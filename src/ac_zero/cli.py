from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ac_zero.agents.base import SolverResult
from ac_zero.agents.greedy import GreedyBestFirstSearch, GreedySolver
from ac_zero.agents.ppo import PPOAgent
from ac_zero.agents.random_agent import RandomLegalActionAgent
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.annotate import AnnotateConfig, annotate, annotation_path
from ac_zero.datasets.candidates import write_candidates
from ac_zero.datasets.generator import generate_solvable
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.hub import DEFAULT_BUCKET, download_dataset, upload_dataset
from ac_zero.datasets.summary import write_dataset_summary
from ac_zero.datasets.validation import validate_dataset
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import create_trainable_model, model_from_json
from ac_zero.moves.universal import MOVE_SET_NAMES
from ac_zero.search.bidirectional import BidirectionalSearch
from ac_zero.search.breadth_first import BreadthFirstSearch
from ac_zero.search.iterative_deepening import IterativeDeepeningConfig, IterativeDeepeningSearch
from ac_zero.search.mcts import UniformMCTS
from ac_zero.search.puct import PUCTMCTS
from ac_zero.system.reporting import CliReporter
from ac_zero.training.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.checkpoint_name import derive_checkpoint_name
from ac_zero.training.hub_checkpoints import PeriodicCheckpointUploader, download_best_checkpoint
from ac_zero.training.pipeline import run_training_pipeline
from ac_zero.training.pipeline_config import TrainingPipelineConfig
from ac_zero.training.smoke import run_smoke_training

# Bare dataset filenames (no directory component) resolve under here, so
# `--input train_rank2.json` lands in `data/generated/train_rank2.json`
# rather than the current working directory.
DATASET_DIR = "data/generated"


def _resolve_dataset_path(value: str) -> str:
    """Anchor a bare dataset filename under ``DATASET_DIR``.

    A value that already carries a directory component (or is empty) is left
    untouched, so explicit paths and the unset ``--output`` default still work.
    """
    if value and Path(value).parent == Path("."):
        return str(Path(DATASET_DIR) / value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Run the AC-Zero command-line interface.

    The current CLI emphasizes deterministic smoke workflows: dataset creation,
    small solver runs, benchmark JSON output, and independent certificate
    verification.
    """

    parser = argparse.ArgumentParser(prog="aczero")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke-test")
    hw = sub.add_parser("hardware")
    hw.add_argument("subcmd", choices=["inspect"])
    ds = sub.add_parser("dataset")
    ds.add_argument(
        "subcmd",
        choices=["grow", "validate", "candidates", "annotate", "upload", "download"],
    )
    ds.add_argument("--config", default="configs/experiments/smoke.yaml")
    ds.add_argument("--output", default="")
    ds.add_argument("--input", default="data/generated/train_rank2.json")
    ds.add_argument(
        "--bucket", default="", help="`upload`/`download`: Hugging Face bucket (default AlphaAC's)"
    )
    ds.add_argument(
        "--remote-name",
        default="",
        help="`upload`/`download`: bucket file name (default: the local basename)",
    )
    ds.add_argument("--total-length-cap", type=int, default=48, help="`grow`: relator length cap")
    ds.add_argument("--max-depth", type=int, default=32, help="`annotate`: max moves per search")
    ds.add_argument(
        "--moveset",
        choices=list(MOVE_SET_NAMES),
        default="universal",
        help="`annotate`: move set to compute distances under",
    )
    ds.add_argument("--rank", type=int, default=2, help="group rank for `grow`")
    ds.add_argument("--target", type=int, default=100, help="`grow`: new groups to add this run")
    ds.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="`grow`: soft wall-clock budget in minutes (0 = run until target/frontier)",
    )
    ds.add_argument(
        "--select",
        choices=["smallest", "weighted-random"],
        default="smallest",
        help="`grow`: which open group to expand next",
    )
    ds.add_argument("--seed", type=int, default=0, help="`grow`: weighted-random selection seed")
    ds.add_argument(
        "--short-bias",
        type=float,
        default=2.0,
        help="`grow`: weighted-random bias toward short groups",
    )
    ds.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="`grow`: dump to disk every N added groups (0 = only at the end)",
    )
    ds.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="`grow`: emit a progress log every N added groups (0 = only start/finish)",
    )
    ds.add_argument(
        "--summary-dir",
        default="data/summaries",
        help="`grow`: directory for the post-generation Markdown summary",
    )
    ds.add_argument(
        "--no-summary",
        action="store_true",
        help="`grow`: skip writing the dataset summary after generation",
    )
    ds.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel worker processes; default 0 uses all CPU cores, 1 stays in-process",
    )
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/experiments/smoke.yaml")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--workers",
        type=int,
        default=None,
        help="self-play worker processes; 0 uses all cores, overrides the config",
    )
    train.add_argument(
        "--upload-checkpoints",
        action="store_true",
        help="push the checkpoint bundle to the HF bucket while training and once at the end",
    )
    train.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="warm-start from the best model already on the HF bucket for this checkpoint name",
    )
    train.add_argument(
        "--checkpoint-name",
        default=None,
        help="HF checkpoint lineage name for up/download (default: derived from the run identity)",
    )
    train.add_argument(
        "--checkpoint-bucket",
        default=None,
        help=f"HF dataset repo to up/download checkpoints from (default {DEFAULT_BUCKET})",
    )
    train.add_argument(
        "--upload-every-hours",
        type=float,
        default=3.0,
        help="hours between checkpoint uploads; the final best model is always pushed",
    )
    bench = sub.add_parser("benchmark")
    bench.add_argument("--config", default="configs/experiments/benchmark_rank2.yaml")
    solve = sub.add_parser("solve")
    solve.add_argument("--presentation", default="data/generated/smoke.json")
    solve.add_argument("--checkpoint", default="")
    solve.add_argument(
        "--agent",
        choices=[
            "greedy",
            "greedy-best-first",
            "breadth-first",
            "bidirectional",
            "iterative-deepening",
            "puct",
            "ppo",
            "uniform-mcts",
        ],
        default="greedy",
    )
    cert = sub.add_parser("certificate")
    cert.add_argument("subcmd", choices=["verify", "render"])
    cert.add_argument("path")
    args = parser.parse_args(argv)

    reporter = CliReporter(args.cmd)
    reporter.info(args.cmd, "running command", {"command": args.cmd})
    try:
        return _dispatch(args, reporter)
    except Exception as exc:
        reporter.error(args.cmd, f"command {args.cmd} failed", exc)
        raise
    finally:
        reporter.close()


def _dispatch(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Route a parsed command to its handler with a shared reporter."""
    if args.cmd == "hardware":
        return _hardware(reporter)
    if args.cmd == "dataset":
        if args.subcmd == "grow":
            return _dataset_grow(args, reporter)
        if args.subcmd == "candidates":
            output = args.output or "data/candidates/standard.json"
            written = write_candidates(output)
            reporter.result_json({"candidates": output, "count": written}, sort_keys=True)
            return 0
        if args.subcmd == "annotate":
            return _dataset_annotate(args, reporter)
        if args.subcmd == "upload":
            return _dataset_upload(args, reporter)
        if args.subcmd == "download":
            return _dataset_download(args, reporter)
        report = validate_dataset(_resolve_dataset_path(args.input))
        reporter.result_json(
            {"ok": report.ok, "entries": report.entries, "errors": report.errors[:20]},
            sort_keys=True,
        )
        if not report.ok:
            reporter.warning("dataset", "dataset failed validation", {"errors": len(report.errors)})
        return 0 if report.ok else 1
    if args.cmd == "train":
        return _train(
            Path(args.config),
            args.seed,
            args.workers,
            reporter,
            upload_checkpoints=args.upload_checkpoints,
            download_checkpoint=args.download_checkpoint,
            checkpoint_name=args.checkpoint_name,
            checkpoint_bucket=args.checkpoint_bucket,
            upload_every_hours=args.upload_every_hours,
        )
    if args.cmd == "benchmark":
        return _benchmark(reporter)
    if args.cmd == "solve":
        return _solve(Path(args.presentation), args.agent, args.checkpoint, reporter)
    if args.cmd == "certificate":
        result = CertificateVerifier().verify_path(args.path)
        if args.subcmd == "render":
            reporter.result_text(result.reason)
        else:
            reporter.result_json(
                {"ok": result.ok, "reason": result.reason, "final_hash": result.final_hash}
            )
        if not result.ok:
            reporter.warning("certificate", "certificate verification failed", {"path": args.path})
        return 0 if result.ok else 1
    if args.cmd == "smoke-test":
        rc = _smoke_train(0, reporter)
        if rc:
            return rc
        benchmark_rc = _benchmark(reporter)
        if benchmark_rc:
            return benchmark_rc
        cert_path = Path("runs/smoke/certificates/example.json")
        return 0 if CertificateVerifier().verify_path(cert_path).ok else 1
    return 2


def _hardware(reporter: CliReporter) -> int:
    """Print a small backend report without requiring JAX to be installed."""
    try:
        import jax  # type: ignore[import-not-found]

        data = {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "default_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }
    except Exception as exc:
        reporter.warning("hardware", "JAX unavailable, defaulting to CPU", {"error": str(exc)})
        data = {"default_backend": "cpu", "warning": f"JAX unavailable: {exc}"}
    reporter.result_json(data, indent=2, sort_keys=True)
    return 0


def _smoke_train(seed: int, reporter: CliReporter) -> int:
    """Run the minimal smoke path and emit a verified fixture certificate.

    This command currently records a checkpoint metadata file and one optimizer
    update metric. It is deliberately small enough for CPU CI and does not claim
    a scientific training result.
    """

    summary = run_smoke_training(seed)
    reporter.result_json(
        {"smoke_training": summary.run_directory, "certificate": summary.certificate_path}
    )
    return 0


def _train(
    config_path: Path,
    seed: int,
    workers: int | None,
    reporter: CliReporter,
    *,
    upload_checkpoints: bool = False,
    download_checkpoint: bool = False,
    checkpoint_name: str | None = None,
    checkpoint_bucket: str | None = None,
    upload_every_hours: float = 3.0,
) -> int:
    """Run the config-driven replay and policy/value training pipeline."""
    config = TrainingPipelineConfig.from_mapping(_load_config(config_path))
    if workers is not None:
        config = replace(config, workers=workers)
    if checkpoint_name:
        config = replace(config, checkpoint_name=checkpoint_name)
    bucket = checkpoint_bucket or DEFAULT_BUCKET
    _ensure_training_dataset(config, reporter)
    if download_checkpoint:
        config = _warm_start_from_hf(config, bucket, reporter)
    callbacks = _training_callbacks(config, upload_checkpoints, bucket, upload_every_hours)
    if callbacks is not None:
        reporter.progress("checkpoint", "uploading checkpoints to bucket", {"bucket": bucket})
    summary = run_training_pipeline(config, seed, callbacks)
    _present_plots(summary.plot_paths, reporter)
    reporter.result_json(
        {
            "training": summary.run_directory,
            "checkpoint": summary.checkpoint_path,
            "certificate": summary.certificate_path,
            "optimizer_updates": summary.optimizer_updates,
            "plots": list(summary.plot_paths),
        },
        sort_keys=True,
    )
    return 0


def _ensure_training_dataset(config: TrainingPipelineConfig, reporter: CliReporter) -> None:
    """Pull the configured self-play dataset from the HF bucket if it is not local.

    A run seeds self-play from `dataset_path` when set; if that file is missing we
    fetch it from the dataset bucket (default AlphaAC's) so `aczero train` works on
    a fresh machine the same way the Kaggle notebook does.
    """
    if not config.dataset_path:
        return
    local = Path(config.dataset_path)
    if local.exists():
        return
    bucket = config.dataset_bucket or DEFAULT_BUCKET
    reporter.progress("dataset", "pulling training dataset from bucket", {"bucket": bucket})
    download_dataset(local, bucket=bucket)


def _warm_start_from_hf(
    config: TrainingPipelineConfig, bucket: str, reporter: CliReporter
) -> TrainingPipelineConfig:
    """Pull this run's best model from the HF bucket and warm-start from it.

    The lineage name is `config.checkpoint_name` when set, else the identity name
    the run would upload under, so download and upload address the same bucket
    prefix. When no checkpoint exists yet the run simply trains from scratch.
    """
    name = config.checkpoint_name or derive_checkpoint_name(config)
    reporter.progress(
        "checkpoint", "pulling best checkpoint from bucket", {"bucket": bucket, "name": name}
    )
    dest = Path(config.run_directory) / "warm_start.json"
    path = download_best_checkpoint(name, dest, bucket=bucket, missing_ok=True)
    if path is None:
        reporter.warning(
            "checkpoint", "no checkpoint on bucket; training from scratch", {"name": name}
        )
        return config
    return replace(config, warm_start=str(path))


def _training_callbacks(
    config: TrainingPipelineConfig,
    upload_checkpoints: bool,
    bucket: str,
    upload_every_hours: float,
) -> CallbackManager | None:
    """Build the callback manager for a run, adding an HF uploader when requested.

    Returns ``None`` when no upload is requested so the pipeline builds its own
    default callbacks; otherwise returns the defaults plus a periodic uploader
    reading the run's bundle directory the pipeline keeps current.
    """
    if not upload_checkpoints:
        return None
    uploader = PeriodicCheckpointUploader(
        Path(config.run_directory) / "model_checkpoint",
        bucket=bucket,
        every_hours=upload_every_hours,
    )
    return default_training_callbacks(config.run_directory, extra=(uploader,))


def _present_plots(plot_paths: Sequence[str], reporter: CliReporter) -> None:
    """Surface the rendered training plots: report them and open in a viewer.

    Each plot path is reported so the run log records where the figures live.
    When stdout is an interactive terminal with a display, the figures are also
    opened in the OS default image viewer; opening is best-effort and never fails
    the command.
    """
    if not plot_paths:
        return
    for path in plot_paths:
        reporter.progress("plots", "training plot ready", {"path": path})
    if not (sys.stdout.isatty() and _has_display()):
        return
    for path in plot_paths:
        try:
            _open_in_viewer(path)
        except OSError as exc:
            reporter.warning(
                "plots", "could not open plot in a viewer", {"path": path, "error": str(exc)}
            )


def _has_display() -> bool:
    """Whether a GUI viewer can plausibly be launched on this platform."""
    if sys.platform.startswith("darwin") or sys.platform.startswith("win"):
        return True
    # On Linux a viewer needs an X11 or Wayland session.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_in_viewer(path: str) -> None:
    """Open one file in the platform's default application."""
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # Windows-only
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _solve(path: Path, agent_name: str, checkpoint: str, reporter: CliReporter) -> int:
    """Load one presentation and solve it with the requested smoke agent."""
    max_moves = 8
    if path.exists() and path.suffix == ".json":
        data = json.loads(path.read_text())
        if "groups" in data:
            groups = data["groups"]
            # Grown datasets carry the trivial root as their first entry; solve the
            # first genuinely non-trivial group instead.
            entry = next((e for e in groups if e.get("source") != "trivial"), groups[0])
            pres = BalancedPresentation.from_letters(int(data["rank"]), entry["relators"])
            max_moves = _max_moves_for_dataset(data, pres)
        else:
            pres = BalancedPresentation.from_json(data)
            max_moves = _max_moves_for_dataset(data, pres)
    else:
        pres = generate_solvable(2, 1, 0).presentation
    env = ACEnvironment(pres, ACEnvironmentConfig(max_moves=max_moves))
    cert_path = Path("runs/solve/certificates/solution.json")
    if agent_name == "greedy":
        result = GreedySolver().solve(env, certificate_path=cert_path, experiment_id="solve")
        if not result.success:
            result = GreedyBestFirstSearch().solve(
                pres,
                env_template=ACEnvironment(pres, ACEnvironmentConfig(max_moves=max_moves)),
                certificate_path=cert_path,
                experiment_id="solve-greedy-best-first-fallback",
            )
    elif agent_name == "greedy-best-first":
        result = GreedyBestFirstSearch().solve(
            pres,
            env_template=env,
            certificate_path=cert_path,
            experiment_id="solve",
        )
    elif agent_name == "breadth-first":
        result = BreadthFirstSearch().solve(
            pres,
            env_template=env,
            certificate_path=cert_path,
            experiment_id="solve",
        )
    elif agent_name == "bidirectional":
        result = BidirectionalSearch().solve(
            pres,
            env_template=env,
            certificate_path=cert_path,
            experiment_id="solve",
        )
    elif agent_name == "iterative-deepening":
        result = IterativeDeepeningSearch().solve(
            pres,
            env_template=env,
            certificate_path=cert_path,
            experiment_id="solve",
        )
    elif agent_name == "puct":
        result = _puct_solve(pres, env, cert_path)
    elif agent_name == "ppo":
        result = _ppo_solve(pres, env, cert_path, checkpoint)
    else:
        mcts = UniformMCTS(8)
        path_ids: list[int] = []
        terminated = False
        for _ in range(8):
            action = mcts.select_action(env)
            path_ids.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        result = GreedySolver()._result(
            pres,
            env.state.presentation,
            tuple(path_ids),
            len(path_ids),
            len(path_ids),
            "goal" if terminated else "horizon",
            terminated,
            cert_path,
            env.config.goal_mode,
            "solve",
            0,
        )
    if not result.success:
        reporter.warning("solve", "agent did not reach the goal", {"agent": agent_name})
    reporter.result_json(
        {
            "success": result.success,
            "moves": list(result.path),
            "best_reduction": result.best_reduction,
            "termination_reason": result.termination_reason,
            "certificate": result.certificate_path,
        },
        sort_keys=True,
    )
    return 0


def _puct_solve(pres: BalancedPresentation, env: ACEnvironment, cert_path: Path) -> SolverResult:
    """Solve greedily by following model-guided PUCT visit counts to termination."""
    mcts = PUCTMCTS(create_trainable_model("residual_mlp", seed=0), StateEncoder())
    path_ids: list[int] = []
    terminated = False
    while len(path_ids) < env.config.max_moves:
        action = mcts.select_action(env)
        path_ids.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return GreedySolver()._result(
        pres,
        env.state.presentation,
        tuple(path_ids),
        len(path_ids),
        len(path_ids),
        "goal" if terminated else "horizon",
        terminated,
        cert_path,
        env.config.goal_mode,
        "solve",
        0,
    )


def _ppo_solve(
    pres: BalancedPresentation, env: ACEnvironment, cert_path: Path, checkpoint: str = ""
) -> SolverResult:
    """Greedily decode a PPO-trained policy to termination, writing a certificate.

    With a checkpoint the policy comes from that trained model; without one it
    falls back to an untrained model so the agent path stays runnable in smoke
    workflows, mirroring how the PUCT smoke solver behaves.
    """
    model = (
        _load_checkpoint_model(checkpoint)
        if checkpoint
        else create_trainable_model("residual_mlp", seed=0)
    )
    agent = PPOAgent(model, StateEncoder())
    path_ids: list[int] = []
    terminated = False
    while len(path_ids) < env.config.max_moves:
        if not any(env.legal_action_mask()):
            break
        action = agent.select_action(env)
        path_ids.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return GreedySolver()._result(
        pres,
        env.state.presentation,
        tuple(path_ids),
        len(path_ids),
        len(path_ids),
        "goal" if terminated else "horizon",
        terminated,
        cert_path,
        env.config.goal_mode,
        "solve",
        0,
    )


def _load_checkpoint_model(checkpoint: str) -> Any:
    """Rebuild a trainable model from a training checkpoint's saved weights."""
    data = json.loads(Path(checkpoint).read_text())
    return model_from_json(data.get("model_state", data))


def _benchmark(reporter: CliReporter) -> int:
    """Run every implemented solver on a shared fixture and write verified results."""
    run = Path("runs/smoke/evaluation")
    certs = Path("runs/smoke/certificates")
    run.mkdir(parents=True, exist_ok=True)
    pres = generate_solvable(2, 2, 0).presentation
    config = ACEnvironmentConfig(max_moves=8, total_length_cap=48)
    rows = []
    for name in (
        "random",
        "greedy",
        "greedy_best_first",
        "breadth_first",
        "bidirectional",
        "iterative_deepening",
        "puct",
        "ppo",
    ):
        env = ACEnvironment(pres, config)
        cert = certs / f"{name}.json"
        rows.append(_solver_row(name, _benchmark_agent(name, pres, env, cert)))
    (run / "benchmark.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    reporter.result_text(run / "benchmark.json")
    return 0


def _benchmark_agent(
    name: str, pres: BalancedPresentation, env: ACEnvironment, cert: Path
) -> SolverResult:
    """Dispatch one benchmark agent to a verified `SolverResult`."""
    if name == "greedy":
        return GreedySolver().solve(env, certificate_path=cert, experiment_id="benchmark")
    if name == "greedy_best_first":
        return GreedyBestFirstSearch().solve(
            pres, env_template=env, certificate_path=cert, experiment_id="benchmark"
        )
    if name == "breadth_first":
        return BreadthFirstSearch().solve(
            pres, env_template=env, certificate_path=cert, experiment_id="benchmark"
        )
    if name == "bidirectional":
        return BidirectionalSearch().solve(
            pres, env_template=env, certificate_path=cert, experiment_id="benchmark"
        )
    if name == "iterative_deepening":
        return IterativeDeepeningSearch(IterativeDeepeningConfig(max_generated=50_000)).solve(
            pres, env_template=env, certificate_path=cert, experiment_id="benchmark"
        )
    if name == "puct":
        return _puct_solve(pres, env, cert)
    if name == "ppo":
        return _ppo_solve(pres, env, cert)
    return _random_rollout(pres, env, cert)


def _random_rollout(pres: BalancedPresentation, env: ACEnvironment, cert: Path) -> SolverResult:
    """Roll out uniformly random legal actions to termination as a weak baseline."""
    agent = RandomLegalActionAgent(random.Random(0))
    path: list[int] = []
    terminated = False
    while len(path) < env.config.max_moves:
        mask = env.legal_action_mask()
        if not any(mask):
            break
        action = agent.select_action(mask)
        path.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return GreedySolver()._result(
        pres,
        env.state.presentation,
        tuple(path),
        len(path),
        len(path),
        "goal" if terminated else "horizon",
        terminated,
        cert,
        env.config.goal_mode,
        "benchmark",
        0,
    )


def _dataset_grow(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Expand the persistent dataset outward from the trivial group by AC moves."""
    path = Path(_resolve_dataset_path(args.output) or _resolve_dataset_path(args.input))
    config = GrowConfig(
        rank=args.rank,
        target=args.target,
        select=args.select,
        seed=args.seed,
        total_length_cap=args.total_length_cap,
        short_bias=args.short_bias,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        time_limit_s=args.minutes * 60 if args.minutes > 0 else None,
    )
    report = grow_dataset(
        path,
        config,
        progress=lambda message, metrics: reporter.progress("grow", message, metrics),
    )
    result: dict[str, Any] = {
        "path": str(path),
        "groups": report.total,
        "added": report.added,
        "expanded": report.expanded,
        "frontier": report.frontier,
        "max_length": report.max_length,
    }
    if not args.no_summary:
        summary_path = write_dataset_summary(path, args.summary_dir)
        reporter.progress("grow", "summary written", {"path": str(summary_path)})
        result["summary"] = str(summary_path)
    reporter.result_json(result, sort_keys=True)
    return 0


def _dataset_upload(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Push a local dataset file to the Hugging Face bucket."""
    local = _resolve_dataset_path(args.input)
    bucket = args.bucket or DEFAULT_BUCKET
    uri = upload_dataset(local, remote_name=args.remote_name or None, bucket=bucket)
    reporter.result_json({"uploaded": local, "uri": uri, "bucket": bucket}, sort_keys=True)
    return 0


def _dataset_download(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Pull a dataset file from the Hugging Face bucket to a local path."""
    local = _resolve_dataset_path(args.output or args.input)
    bucket = args.bucket or DEFAULT_BUCKET
    path = download_dataset(local, remote_name=args.remote_name or None, bucket=bucket)
    reporter.result_json({"downloaded": str(path), "bucket": bucket}, sort_keys=True)
    return 0


def _dataset_annotate(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Annotate a group dataset with distances under one move set."""
    input_path = _resolve_dataset_path(args.input)
    config = AnnotateConfig(
        moveset=args.moveset,
        max_depth=args.max_depth,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )
    report = annotate(
        input_path,
        config,
        progress=lambda message, metrics: reporter.progress("annotate", message, metrics),
    )
    reporter.result_json(
        {
            "input": input_path,
            "output": str(annotation_path(input_path, args.moveset)),
            "moveset": report.moveset,
            "total": report.total,
            "reached_origin": report.reached_origin,
            "with_shorter": report.with_shorter,
            "computed": report.computed,
            "max_distance_to_origin": report.max_distance_to_origin,
        },
        sort_keys=True,
    )
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    """Load a YAML experiment config, returning defaults when the file is absent."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(data)


def _max_moves_for_dataset(data: dict[str, Any], pres: BalancedPresentation) -> int:
    """Choose a solve horizon large enough for generated scramble fixtures."""
    depth = _depth_from_mapping(data.get("provenance"))
    depth = depth or _depth_from_mapping(pres.provenance)
    if depth is None:
        return 8
    return max(8, depth * 3)


def _depth_from_mapping(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    depth = value.get("depth")
    if depth is None:
        return None
    return int(depth)


def _solver_row(name: str, result: SolverResult) -> dict[str, object]:
    """Convert a `SolverResult` into a benchmark-report row."""
    return {
        "agent": name,
        "verified_success": result.success,
        "path": list(result.path),
        "best_reduction": result.best_reduction,
        "expanded_nodes": result.expanded_nodes,
        "generated_nodes": result.generated_nodes,
        "peak_frontier_size": result.peak_frontier_size,
        "termination_reason": result.termination_reason,
        "certificate": result.certificate_path,
        **result.metrics,
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
