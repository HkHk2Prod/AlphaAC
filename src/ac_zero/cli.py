from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, TypedDict

import yaml  # type: ignore[import-untyped]

from ac_zero.agents.base import SolverResult
from ac_zero.agents.greedy import GreedyBestFirstSearch, GreedySolver
from ac_zero.agents.random_agent import RandomLegalActionAgent
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.candidates import write_candidates
from ac_zero.datasets.generator import generate_solvable, write_dataset
from ac_zero.datasets.update import (
    BreadthFirstStrategy,
    GreedyBestFirstStrategy,
    SearchStrategy,
    improve_dataset,
)
from ac_zero.datasets.validation import validate_dataset
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import create_trainable_model
from ac_zero.search.breadth_first import BreadthFirstSearch
from ac_zero.search.iterative_deepening import IterativeDeepeningConfig, IterativeDeepeningSearch
from ac_zero.search.mcts import UniformMCTS
from ac_zero.search.puct import PUCTMCTS
from ac_zero.system.reporting import CliReporter
from ac_zero.training.pipeline import TrainingPipelineConfig, run_training_pipeline
from ac_zero.training.smoke import run_smoke_training


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
    ds.add_argument("subcmd", choices=["generate", "validate", "candidates", "improve"])
    ds.add_argument("--config", default="configs/experiments/smoke.yaml")
    ds.add_argument("--output", default="")
    ds.add_argument("--input", default="data/generated/train_rank2.json")
    ds.add_argument("--search", choices=["bfs", "greedy-best-first", "all"], default="all")
    ds.add_argument("--max-moves", type=int, default=12)
    ds.add_argument("--total-length-cap", type=int, default=48)
    ds.add_argument("--max-difficulty", type=int, default=8)
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/experiments/smoke.yaml")
    train.add_argument("--seed", type=int, default=0)
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
            "iterative-deepening",
            "puct",
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
        if args.subcmd == "generate":
            output = args.output or "data/generated/smoke.json"
            params = _dataset_generation_params(Path(args.config))
            write_dataset(output, **params)
            reporter.result_text(output)
            return 0
        if args.subcmd == "candidates":
            output = args.output or "data/candidates/standard.json"
            written = write_candidates(output)
            reporter.result_json({"candidates": output, "count": written}, sort_keys=True)
            return 0
        if args.subcmd == "improve":
            return _dataset_improve(args, reporter)
        report = validate_dataset(args.input)
        reporter.result_json(
            {"ok": report.ok, "instances": report.instances, "errors": report.errors[:20]},
            sort_keys=True,
        )
        if not report.ok:
            reporter.warning("dataset", "dataset failed validation", {"errors": len(report.errors)})
        return 0 if report.ok else 1
    if args.cmd == "train":
        return _train(Path(args.config), args.seed, reporter)
    if args.cmd == "benchmark":
        return _benchmark(reporter)
    if args.cmd == "solve":
        return _solve(Path(args.presentation), args.agent, reporter)
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


def _train(config_path: Path, seed: int, reporter: CliReporter) -> int:
    """Run the config-driven replay and policy/value training pipeline."""
    config = TrainingPipelineConfig.from_mapping(_load_config(config_path))
    summary = run_training_pipeline(config, seed)
    reporter.result_json(
        {
            "training": summary.run_directory,
            "checkpoint": summary.checkpoint_path,
            "certificate": summary.certificate_path,
            "optimizer_updates": summary.optimizer_updates,
        },
        sort_keys=True,
    )
    return 0


def _solve(path: Path, agent_name: str, reporter: CliReporter) -> int:
    """Load one presentation and solve it with the requested smoke agent."""
    max_moves = 8
    if path.exists() and path.suffix == ".json":
        data = json.loads(path.read_text())
        if "instances" in data:
            pres = BalancedPresentation.from_json(data["instances"][0])
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
    elif agent_name == "iterative-deepening":
        result = IterativeDeepeningSearch().solve(
            pres,
            env_template=env,
            certificate_path=cert_path,
            experiment_id="solve",
        )
    elif agent_name == "puct":
        result = _puct_solve(pres, env, cert_path)
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
        "iterative_deepening",
        "puct",
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
    if name == "iterative_deepening":
        return IterativeDeepeningSearch(IterativeDeepeningConfig(max_generated=50_000)).solve(
            pres, env_template=env, certificate_path=cert, experiment_id="benchmark"
        )
    if name == "puct":
        return _puct_solve(pres, env, cert)
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


def _dataset_improve(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Search dataset entries and merge in any shorter or newly found solutions."""
    strategies: list[SearchStrategy] = []
    if args.search in ("bfs", "all"):
        strategies.append(
            BreadthFirstStrategy(max_moves=args.max_moves, total_length_cap=args.total_length_cap)
        )
    if args.search in ("greedy-best-first", "all"):
        strategies.append(GreedyBestFirstStrategy())
    output = args.output or args.input
    report = improve_dataset(
        args.input,
        strategies=strategies,
        output=output,
        max_difficulty=args.max_difficulty,
    )
    reporter.result_json(
        {
            "output": output,
            "total": report.total,
            "duplicates_merged": report.duplicates_merged,
            "searched": report.searched,
            "solved": report.solved,
            "improved": report.improved,
            "proved_optimal": report.proved_optimal,
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


class _DatasetParams(TypedDict):
    rank: int
    count: int
    depth: int
    seed: int
    min_total_length: int
    min_relator_length: int


def _dataset_generation_params(config_path: Path) -> _DatasetParams:
    """Read dataset generation parameters from an experiment config."""
    data = _load_config(config_path)
    dataset = data.get("dataset", {})
    if dataset is None:
        dataset = {}
    if not isinstance(dataset, dict):
        raise ValueError("dataset config must be a mapping")
    return {
        "rank": int(data.get("rank", 2)),
        "count": int(dataset.get("count", data.get("count", 3))),
        "depth": int(dataset.get("depth", data.get("depth", 3))),
        "seed": int(data.get("seed", 0)),
        "min_total_length": int(dataset.get("min_total_length", data.get("min_total_length", 0))),
        "min_relator_length": int(
            dataset.get("min_relator_length", data.get("min_relator_length", 0))
        ),
    }


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
