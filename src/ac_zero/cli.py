from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ac_zero.agents.base import SolverResult
from ac_zero.agents.greedy import GreedyBestFirstSearch, GreedySolver
from ac_zero.agents.ppo import PPOAgent
from ac_zero.agents.random_agent import RandomLegalActionAgent
from ac_zero.algebra.presentation import BalancedPresentation
from ac_zero.benchmarks.catalog import DEFAULT_MAX_W_LENGTH, catalog_name
from ac_zero.benchmarks.commands import (
    DEFAULT_CATALOG_DIR,
    create_catalog,
    load_benchmark_config,
    run_benchmark,
)
from ac_zero.benchmarks.config import BenchmarkConfig
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.annotate import AnnotateConfig, annotate, annotation_path
from ac_zero.datasets.ball import BallConfig, ball_groups_path, grow_ball
from ac_zero.datasets.candidates import write_candidates
from ac_zero.datasets.generator import generate_solvable
from ac_zero.datasets.grow import GrowConfig, grow_dataset
from ac_zero.datasets.hub import DEFAULT_BUCKET, download_dataset, remote_size, upload_dataset
from ac_zero.datasets.instance_store import InstanceStore
from ac_zero.datasets.publish import publish_to_bucket
from ac_zero.datasets.remote_paths import dataset_remote_name
from ac_zero.datasets.split import SplitConfig, split_is_current, split_path, write_split
from ac_zero.datasets.summary import write_annotation_summary, write_dataset_summary
from ac_zero.datasets.supervised_store import SupervisedStore
from ac_zero.datasets.validation import validate_dataset
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import (
    checkpoint_format_version,
    create_trainable_model,
    model_from_json,
)
from ac_zero.models.trainable import MODEL_FORMAT_VERSION
from ac_zero.moves.universal import MOVE_SET_NAMES
from ac_zero.search.bidirectional import BidirectionalSearch
from ac_zero.search.breadth_first import BreadthFirstSearch
from ac_zero.search.iterative_deepening import IterativeDeepeningConfig, IterativeDeepeningSearch
from ac_zero.search.mcts import UniformMCTS
from ac_zero.search.puct import PUCTMCTS
from ac_zero.system.reporting import CliReporter
from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
from ac_zero.training.checkpointing.hub_checkpoints import (
    ARCHIVE_PREFIX,
    PeriodicCheckpointUploader,
    archive_checkpoint_lineage,
    download_best_checkpoint,
)
from ac_zero.training.logging.callbacks import CallbackManager, default_training_callbacks
from ac_zero.training.logging.events import Verbosity
from ac_zero.training.pipeline.pipeline import run_training_pipeline
from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig
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
        choices=[
            "grow",
            "ball",
            "validate",
            "candidates",
            "annotate",
            "split",
            "labels",
            "upload",
            "download",
        ],
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
        help="`upload`/`download`: explicit bucket path (default: the dataset's folder, "
        "derived from the filename, e.g. datasets/rank2/rel-48/)",
    )
    ds.add_argument(
        "--max-relator-length",
        type=int,
        default=0,
        help="`grow`/`ball`: longest relator a group may carry; a move that would "
        "overshoot it is one the environment masks, so the dataset holds exactly the "
        "groups a model of that encoder capacity can reach (0 = unbounded)",
    )
    ds.add_argument(
        "--max-depth", type=int, default=32, help="`annotate`: max moves per search (0 = unbounded)"
    )
    ds.add_argument(
        "--moveset",
        choices=list(MOVE_SET_NAMES),
        default="universal",
        help="`annotate`/`ball`: move set to compute distances under",
    )
    ds.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="`split`: share of groups held out for validation (default 0.1)",
    )
    ds.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="`split`: share of groups held out for the final test report (default 0.1)",
    )
    ds.add_argument(
        "--split-salt",
        default=SplitConfig().salt,
        help="`split`: changing this reshuffles every group into a new split, which "
        "invalidates any model already scored against the old one",
    )
    ds.add_argument("--rank", type=int, default=2, help="group rank for `grow`/`ball`")
    ds.add_argument(
        "--target", type=int, default=100, help="`grow`/`ball`: new groups to add this run"
    )
    ds.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="`grow`/`ball`: soft wall-clock budget in minutes (0 = run until target/frontier)",
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
        "--checkpoint-hours",
        type=float,
        default=4.0,
        help="`ball`: every N hours, rewrite both documents and push them to the bucket "
        "(0 = only at the end). This is how much progress an interruption may cost: a "
        "checkpoint left on a disk that dies with the machine buys nothing, so the write "
        "and the upload are one event",
    )
    ds.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="`grow`: emit a progress log every N added groups (0 = only start/finish)",
    )
    ds.add_argument(
        "--no-atomic-checkpoint",
        action="store_true",
        help="`ball`: write each checkpoint in place instead of via an atomic temp copy, "
        "halving the peak disk a rewrite needs. A torn write corrupts the local file, so "
        "use it only when a durable copy is pushed elsewhere -- as the scheduler does to "
        "the bucket resume pulls from",
    )
    ds.add_argument(
        "--summary-dir",
        default="data/summaries",
        help="`grow`/`annotate`: directory for the post-run Markdown summary",
    )
    ds.add_argument(
        "--no-summary",
        action="store_true",
        help="`grow`/`annotate`: skip writing (and uploading) the Markdown summary",
    )
    ds.add_argument(
        "--no-upload",
        action="store_true",
        help="`grow`/`annotate`: skip pushing the dataset/annotation file and its summary "
        "to the Hugging Face bucket (uploads are on by default)",
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
        "--minutes",
        type=float,
        default=0.0,
        help="soft wall-clock budget in minutes; the run stops at the next iteration boundary "
        "past it and still writes its checkpoint, plots, and summary (0 = run all iterations)",
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
        "--start-fresh",
        action="store_true",
        help="abandon this checkpoint name's history: move its bucket tree under "
        "model_checkpoints/_archive/ and train from the pretrained checkpoint (or from "
        "scratch when the config names none) instead of resuming",
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
    train.add_argument(
        "--showcase-every-hours",
        type=float,
        default=None,
        help="hours between self-play showcases: one episode played with the current weights "
        "and printed move by move (default 3, matching the upload cadence; 0 disables)",
    )
    train.add_argument(
        "--self-generated",
        action="store_true",
        help="seed self-play from random scrambles instead of the HF group dataset (the default); "
        "ignores any dataset.path in the config",
    )
    train.add_argument(
        "--force-download-dataset",
        action="store_true",
        help="re-pull the dataset groups and annotations from the bucket even if present locally",
    )
    train.add_argument(
        "--verbosity",
        choices=["quiet", "summary", "verbose"],
        default=None,
        help="terminal detail: 'verbose' (per-event lines + live graphs), 'summary' (one line "
        "per iteration + final graph, the default), or 'quiet' (start/stop + warnings); "
        "overrides the config. The JSONL log and graph files always record everything",
    )
    bench = sub.add_parser("benchmark")
    # `solvers` (the default) is the cross-solver regression on a fixture;
    # `create`/`run` are the AK/MS model benchmark.
    bench.add_argument("subcmd", nargs="?", choices=["solvers", "create", "run"], default="solvers")
    bench.add_argument("--catalog", default="", help="benchmark catalog file to write or score")
    bench.add_argument("--max-relator-length", type=int, default=48)
    bench.add_argument("--max-w-length", type=int, default=DEFAULT_MAX_W_LENGTH)
    bench.add_argument("--config", default="", help="YAML of benchmark budgets")
    bench.add_argument("--checkpoint", default="", help="local checkpoint file to evaluate")
    bench.add_argument("--checkpoint-name", default="", help="HF lineage whose best model to pull")
    bench.add_argument("--bucket", default=DEFAULT_BUCKET)
    bench.add_argument("--run-id", default="")
    bench.add_argument("--minutes", type=float, default=0.0, help="wall-clock cap (0 = unlimited)")
    bench.add_argument("--upload", action="store_true", help="`run`: publish under benchmarks/")
    bench.add_argument(
        "--no-upload",
        action="store_true",
        help="`create`: keep the catalog local instead of publishing it to benchmark_datasets/",
    )
    bench.add_argument("--output", default="")
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
        if args.subcmd == "ball":
            return _dataset_ball(args, reporter)
        if args.subcmd == "candidates":
            output = args.output or "data/candidates/standard.json"
            written = write_candidates(output)
            reporter.result_json({"candidates": output, "count": written}, sort_keys=True)
            return 0
        if args.subcmd == "annotate":
            return _dataset_annotate(args, reporter)
        if args.subcmd == "split":
            return _dataset_split(args, reporter)
        if args.subcmd == "labels":
            return _dataset_labels(args, reporter)
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
            minutes=args.minutes,
            upload_checkpoints=args.upload_checkpoints,
            download_checkpoint=args.download_checkpoint,
            start_fresh=args.start_fresh,
            checkpoint_name=args.checkpoint_name,
            checkpoint_bucket=args.checkpoint_bucket,
            upload_every_hours=args.upload_every_hours,
            showcase_every_hours=args.showcase_every_hours,
            self_generated=args.self_generated,
            force_download_dataset=args.force_download_dataset,
            verbosity=args.verbosity,
        )
    if args.cmd == "benchmark":
        if args.subcmd == "create":
            reporter.result_json(_benchmark_create(args), sort_keys=True)
            return 0
        if args.subcmd == "run":
            reporter.result_json(_benchmark_run(args), sort_keys=True)
            return 0
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
    minutes: float = 0.0,
    upload_checkpoints: bool = False,
    download_checkpoint: bool = False,
    start_fresh: bool = False,
    checkpoint_name: str | None = None,
    checkpoint_bucket: str | None = None,
    upload_every_hours: float = 3.0,
    showcase_every_hours: float | None = None,
    self_generated: bool = False,
    force_download_dataset: bool = False,
    verbosity: str | None = None,
) -> int:
    """Run the config-driven replay and policy/value training pipeline."""
    config = TrainingPipelineConfig.from_mapping(_load_config(config_path))
    if workers is not None:
        config = replace(config, workers=workers)
    if verbosity is not None:
        config = replace(config, verbosity=Verbosity.parse(verbosity))
    if showcase_every_hours is not None:
        config = replace(config, showcase_every_hours=showcase_every_hours)
    if minutes > 0:
        config = replace(config, time_limit_s=minutes * 60)
    if checkpoint_name:
        config = replace(config, checkpoint_name=checkpoint_name)
    # Seed self-play from the HF group dataset by default (deriving the rank/moveset
    # file names when the config leaves them unset); `--self-generated` opts back
    # into random scrambles by clearing the dataset paths.
    if self_generated:
        config = replace(config, dataset_path=None, dataset_annotations_path=None)
    else:
        config = _seed_from_default_dataset(config)
    bucket = checkpoint_bucket or DEFAULT_BUCKET
    _ensure_training_dataset(config, reporter, force=force_download_dataset)
    # `--start-fresh` pulls in this path on its own: it has a lineage to archive and a
    # pretrained checkpoint to re-seed from, so it must never silently no-op when the
    # caller omitted `--download-checkpoint`.
    if download_checkpoint or start_fresh:
        config = _warm_start_from_hf(config, bucket, reporter, start_fresh=start_fresh)
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
            "iterations": summary.iterations,
            "optimizer_updates": summary.optimizer_updates,
            "plots": list(summary.plot_paths),
        },
        sort_keys=True,
    )
    return 0


def _seed_from_default_dataset(config: TrainingPipelineConfig) -> TrainingPipelineConfig:
    """Point the run at the default dataset files for its rank, bound, and move set.

    The default is the closest-first ``ball_rank{rank}_rel{max_relator_tokens}`` ball and
    its companion annotations: every distance in it is a proven optimum, and every group
    within its ``complete_depth`` of the origin is there -- neither of which the
    length-first ``train_rank{rank}`` dataset can claim. The run's ``max_relator_tokens``
    picks the file because it is the bound the ball was grown under; a run that changes
    its encoder capacity trains on a different ball, not a filtered view of the same one.
    Explicit ``dataset.path``/``dataset.annotations`` in the config win.
    """
    default_groups = ball_groups_path(DATASET_DIR, config.rank, config.max_relator_tokens)
    groups = config.dataset_path or str(default_groups)
    annotations = config.dataset_annotations_path or str(annotation_path(groups, config.moveset))
    split = config.dataset_split_path or str(split_path(groups))
    return replace(
        config,
        dataset_path=groups,
        dataset_annotations_path=annotations,
        dataset_split_path=split,
    )


def _ensure_training_dataset(
    config: TrainingPipelineConfig, reporter: CliReporter, *, force: bool = False
) -> None:
    """Pull the configured self-play dataset (groups + annotations) from the bucket.

    A run seeds self-play from `dataset_path` when set; its companion annotations
    carry the per-group distances. Each file is fetched when missing locally so `aczero
    train` works on a fresh machine the same way the Kaggle notebook does; `force`
    re-downloads both even when they already exist, refreshing a stale copy.
    """
    if not config.dataset_path:
        return
    bucket = config.dataset_bucket or DEFAULT_BUCKET
    # Supervised runs read the whole ball plus a split that must cover every group, so
    # they provision differently: refresh each file against the bucket by size and rebuild
    # the split whenever it no longer matches the dataset (see `_provision_supervised_dataset`).
    if config.agent == "supervised":
        _provision_supervised_dataset(config, bucket, reporter)
        return
    groups = Path(config.dataset_path)
    if force or not groups.exists():
        reporter.progress("dataset", "pulling groups from bucket", {"bucket": bucket})
        download_dataset(groups, bucket=bucket)
    if not config.dataset_annotations_path:
        return
    annotations = Path(config.dataset_annotations_path)
    if force or not annotations.exists():
        reporter.progress("dataset", "pulling annotations from bucket", {"bucket": bucket})
        if download_dataset(annotations, bucket=bucket, missing_ok=True) is None:
            reporter.warning(
                "dataset", "annotations absent from bucket", {"name": annotations.name}
            )


def _provision_supervised_dataset(
    config: TrainingPipelineConfig, bucket: str, reporter: CliReporter
) -> None:
    """Refresh the supervised dataset from the bucket and (re)build its split.

    A supervised run trains on the whole ball and evaluates on its split, so a stale
    local copy would train on old data or a split that no longer covers every group. Each
    file is compared to the bucket by byte size and re-pulled only when it differs --
    freshness without re-downloading hundreds of megabytes that already match -- and the
    split is regenerated whenever it is missing or was built from a different dataset. The
    split is a local artifact (it is not kept in the bucket), so it is generated here
    rather than downloaded. Every decision is logged so the run's opening lines show
    exactly what was refreshed.
    """
    groups = Path(config.dataset_path or "")
    reporter.progress("dataset", "provisioning supervised dataset", {"bucket": bucket})
    _sync_bucket_file(groups, bucket, reporter, required=True)
    if config.dataset_annotations_path:
        _sync_bucket_file(Path(config.dataset_annotations_path), bucket, reporter, required=True)
    _ensure_dataset_split(groups, reporter)


def _sync_bucket_file(local: Path, bucket: str, reporter: CliReporter, *, required: bool) -> None:
    """Download ``local`` from the bucket only when the local copy differs from it by size."""
    reporter.progress("dataset", "checking bucket for current size", {"name": local.name})
    remote = remote_size(dataset_remote_name(local.name), bucket=bucket)
    local_bytes = local.stat().st_size if local.exists() else None
    if remote is None:
        if local_bytes is None:
            if required:
                raise FileNotFoundError(
                    f"{local.name} is absent both locally and from bucket {bucket!r}; "
                    "grow or download the dataset before training on it"
                )
            reporter.warning("dataset", "absent locally and from bucket", {"name": local.name})
        else:
            reporter.progress("dataset", "not in bucket; keeping local copy", {"name": local.name})
        return
    if local_bytes == remote:
        reporter.progress(
            "dataset",
            "local copy matches bucket; skipping download",
            {"name": local.name, "bytes": remote},
        )
        return
    reporter.progress(
        "dataset",
        "local copy differs from bucket; downloading",
        {"name": local.name, "local_bytes": local_bytes, "remote_bytes": remote},
    )
    download_dataset(local, bucket=bucket)


def _ensure_dataset_split(groups: Path, reporter: CliReporter) -> None:
    """Keep the split beside ``groups`` when it matches the dataset, else regenerate it."""
    split = split_path(groups)
    current, reason = split_is_current(groups)
    if current:
        reporter.progress("dataset", "split matches dataset; keeping it", {"name": split.name})
        return
    reporter.progress("dataset", f"regenerating split: {reason}", {"name": split.name})
    report = write_split(groups, SplitConfig())
    reporter.progress(
        "dataset",
        "split written",
        {"train": report.train, "val": report.val, "test": report.test, "total": report.total},
    )


def _reject_stale_checkpoint(path: Path, name: str, remedy: str) -> None:
    """Fail the run now if the pulled checkpoint predates the current model format.

    The load itself happens deep inside the pipeline, after the dataset download and the
    supervised sidecar build -- half an hour into a Kaggle slot. The version is one field
    of the payload, so checking it here turns that into an immediate, actionable error.
    """
    version = checkpoint_format_version(json.loads(path.read_text(encoding="utf-8")))
    if version != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {name!r} on the bucket is model format v{version}, but this code "
            f"writes and loads v{MODEL_FORMAT_VERSION}; it cannot seed this run -- {remedy}."
        )


def _warm_start_from_hf(
    config: TrainingPipelineConfig,
    bucket: str,
    reporter: CliReporter,
    *,
    start_fresh: bool = False,
) -> TrainingPipelineConfig:
    """Pull this run's best model from the HF bucket and warm-start from it.

    The lineage name is `config.checkpoint_name` when set, else the identity name
    the run would upload under, so download and upload address the same bucket
    prefix. When this task has no checkpoint of its own yet -- its first run -- and it
    names a `pretrained_checkpoint`, seed from that supervised-pretrained model instead;
    every later run finds this task's own checkpoint above, so the RL checkpoint always
    wins once it exists and the pretrained model only ever seeds run one. With neither on
    the bucket the run trains from scratch.

    `start_fresh` restarts the lineage: the existing tree is archived first, which leaves
    the name empty and drops the run into exactly the "first run" branch below -- so a
    fresh start seeds from the pretrained checkpoint on a pretrained task and from zero on
    a scratch one, with no separate code path deciding which.
    """
    name = config.checkpoint_name or derive_checkpoint_name(config)
    if start_fresh:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        moved = archive_checkpoint_lineage(name, stamp, bucket=bucket)
        reporter.progress(
            "checkpoint",
            "start_fresh: archived the existing checkpoint lineage",
            {"name": name, "files": moved, "archive": f"{ARCHIVE_PREFIX}/{name}/{stamp}/"},
        )
    reporter.progress(
        "checkpoint", "pulling best checkpoint from bucket", {"bucket": bucket, "name": name}
    )
    dest = Path(config.run_directory) / "warm_start.json"
    path = download_best_checkpoint(name, dest, bucket=bucket, missing_ok=True)
    if path is not None:
        _reject_stale_checkpoint(
            path, name, f"re-run this task with --start-fresh to archive {name!r} and start over"
        )
        return replace(config, warm_start=str(path))
    if config.pretrained_checkpoint:
        pretrained = config.pretrained_checkpoint
        reporter.progress(
            "checkpoint",
            "no run checkpoint yet; seeding first run from the pretrained checkpoint",
            {"bucket": bucket, "pretrained": pretrained},
        )
        pretrained_dest = Path(config.run_directory) / "pretrained_warm_start.json"
        pretrained_path = download_best_checkpoint(
            pretrained, pretrained_dest, bucket=bucket, missing_ok=True
        )
        if pretrained_path is not None:
            _reject_stale_checkpoint(
                pretrained_path,
                pretrained,
                "re-run supervised pretraining for that lineage to publish a current checkpoint",
            )
            return replace(config, warm_start=str(pretrained_path))
        reporter.warning(
            "checkpoint",
            "pretrained checkpoint absent from bucket; training from scratch",
            {"name": pretrained},
        )
        return config
    reporter.warning("checkpoint", "no checkpoint on bucket; training from scratch", {"name": name})
    return config


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
    return default_training_callbacks(
        config.run_directory, verbosity=config.verbosity, extra=(uploader,)
    )


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


def _benchmark_create(args: argparse.Namespace) -> dict[str, Any]:
    """Enumerate the AK/MS catalog under the given bounds."""
    return create_catalog(
        max_relator_length=args.max_relator_length,
        max_w_length=args.max_w_length,
        output=args.catalog or args.output,
        upload=not args.no_upload,
        bucket=args.bucket,
    )


def _benchmark_run(args: argparse.Namespace) -> dict[str, Any]:
    """Score one checkpoint against a catalog, optionally publishing the result."""
    name = catalog_name(args.max_relator_length, args.max_w_length)
    catalog = args.catalog or str(Path(DEFAULT_CATALOG_DIR) / f"{name}.json")
    # Only flags the caller actually set override the config file; an unset flag
    # carries an empty default that would otherwise erase the configured value.
    overrides: dict[str, Any] = {"catalog_path": catalog, "bucket": args.bucket}
    if args.checkpoint:
        overrides["checkpoint_path"] = args.checkpoint
    if args.checkpoint_name:
        overrides["checkpoint_name"] = args.checkpoint_name
    if args.minutes:
        overrides["max_total_minutes"] = args.minutes
    config = BenchmarkConfig.from_mapping({**load_benchmark_config(args.config), **overrides})
    run_id = args.run_id or f"{int(time.time())}"
    payload = run_benchmark(config, run_id=run_id, upload=args.upload, output=args.output)
    # The solved ids are the record, not the terminal summary: on a full catalog
    # they run to thousands. They stay in the written file and in the HF summary.
    return {**payload, "solved_ids": payload["solved_ids"][:20]}


def _benchmark(reporter: CliReporter) -> int:
    """Run every implemented solver on a shared fixture and write verified results."""
    run = Path("runs/smoke/evaluation")
    certs = Path("runs/smoke/certificates")
    run.mkdir(parents=True, exist_ok=True)
    pres = generate_solvable(2, 2, 0).presentation
    config = ACEnvironmentConfig(max_moves=8)
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


def _publish_dataset_artifact(
    args: argparse.Namespace,
    reporter: CliReporter,
    *,
    phase: str,
    data_path: Path,
    summary_writer: Any,
    result: dict[str, Any],
) -> None:
    """Write the Markdown summary and push the data file and summary to the HF bucket.

    ``summary_writer`` is ``write_dataset_summary`` (grow) or
    ``write_annotation_summary`` (annotate). ``--no-summary`` skips the report;
    ``--no-upload`` skips the bucket push. Upload failures (missing
    ``huggingface_hub``, absent ``HF_TOKEN``, or a hub error) are reported per file
    as a warning rather than failing the run, so offline runs still succeed.
    Mutates ``result`` in place.
    """
    published = publish_to_bucket(
        data_path,
        summary_writer=None if args.no_summary else summary_writer,
        summary_dir=args.summary_dir,
        bucket=args.bucket or DEFAULT_BUCKET,
        upload=not args.no_upload,
    )
    if published.summary_path is not None:
        reporter.progress(phase, "summary written", {"path": str(published.summary_path)})
        result["summary"] = str(published.summary_path)
    for outcome in published.outcomes:
        if not outcome.ok:
            reporter.warning(
                phase, "HF upload skipped", {"file": outcome.remote_name, "error": outcome.error}
            )
    if published.uploaded_uris:
        result["uploaded"] = published.uploaded_uris
        reporter.progress(phase, "uploaded to HF bucket", {"files": len(published.uploaded_uris)})


def _upload_ball_checkpoint(
    args: argparse.Namespace, reporter: CliReporter, groups_path: Path, annotations_path: Path
) -> None:
    """Push a mid-run checkpoint of both ball documents to the bucket.

    No summary: that is an end-of-run report, and rendering one every few hours over a
    growing dataset would cost more than the checkpoint it accompanies. Uploads never
    raise -- a hub blip must not kill a run that has hours of expansion behind it -- so a
    failure is a warning and the next checkpoint tries again.
    """
    if args.no_upload:
        return
    published = (
        publish_to_bucket(groups_path, bucket=args.bucket or DEFAULT_BUCKET).outcomes
        + publish_to_bucket(annotations_path, bucket=args.bucket or DEFAULT_BUCKET).outcomes
    )
    for outcome in published:
        if not outcome.ok:
            reporter.warning(
                "ball", "HF upload skipped", {"file": outcome.remote_name, "error": outcome.error}
            )
    uploaded = [outcome.remote_name for outcome in published if outcome.ok]
    if uploaded:
        reporter.progress("ball", "checkpoint pushed to HF bucket", {"files": len(uploaded)})


def _dataset_grow(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Expand the persistent dataset outward from the trivial group by AC moves."""
    path = Path(_resolve_dataset_path(args.output) or _resolve_dataset_path(args.input))
    config = GrowConfig(
        rank=args.rank,
        target=args.target,
        select=args.select,
        seed=args.seed,
        max_relator_length=args.max_relator_length,
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
    _publish_dataset_artifact(
        args,
        reporter,
        phase="grow",
        data_path=path,
        summary_writer=write_dataset_summary,
        result=result,
    )
    reporter.result_json(result, sort_keys=True)
    return 0


def _dataset_ball(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Grow the ball around the trivial group closest first, under one move set.

    Unlike `grow`, this expands by the *inverses* of the move set's moves in
    breadth-first order, so the distances it writes are proven optima rather than
    upper bounds, and every group within `complete_depth` of the origin is present.
    The distances are emitted as the companion annotation file, so no `annotate` pass
    follows; both files are summarized and pushed to the bucket.

    Every ``--checkpoint-hours`` the run rewrites both documents *and pushes them*. On a
    machine that can be taken away mid-run -- a Kaggle session, a spot instance -- a
    checkpoint left on a disk that dies with the container buys nothing, so the local
    write and the bucket push are one event: the interval is how much progress an
    interruption may cost, and nothing else.
    """
    groups_path = Path(
        _resolve_dataset_path(args.output)
        or ball_groups_path(DATASET_DIR, args.rank, args.max_relator_length)
    )
    annotations_path = annotation_path(groups_path, args.moveset)
    config = BallConfig(
        rank=args.rank,
        moveset=args.moveset,
        target=args.target,
        max_relator_length=args.max_relator_length,
        workers=args.workers,
        checkpoint_hours=args.checkpoint_hours,
        log_every=args.log_every,
        time_limit_s=args.minutes * 60 if args.minutes > 0 else None,
        atomic_checkpoint=not args.no_atomic_checkpoint,
    )

    def on_progress(message: str, metrics: dict[str, Any]) -> None:
        reporter.progress("ball", message, metrics)
        if message == "checkpoint":
            _upload_ball_checkpoint(args, reporter, groups_path, annotations_path)

    report = grow_ball(groups_path, config, progress=on_progress)
    result: dict[str, Any] = {
        "path": str(groups_path),
        "annotations": str(annotations_path),
        "moveset": args.moveset,
        "max_relator_length": args.max_relator_length,
        "groups": report.total,
        "added": report.added,
        "expanded": report.expanded,
        "complete_depth": report.complete_depth,
        "max_distance": report.max_distance,
        "max_length": report.max_length,
    }
    # Two artifacts, each with its own summary: the groups and the distances that
    # label them. They are published separately but reported as one run.
    summaries: list[str] = []
    uploaded: list[str] = []
    for data_path, summary_writer in (
        (groups_path, write_dataset_summary),
        (annotations_path, write_annotation_summary),
    ):
        published: dict[str, Any] = {}
        _publish_dataset_artifact(
            args,
            reporter,
            phase="ball",
            data_path=data_path,
            summary_writer=summary_writer,
            result=published,
        )
        summaries += [published["summary"]] if "summary" in published else []
        uploaded += published.get("uploaded", [])
    result["summaries"] = summaries
    result["uploaded"] = uploaded
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
    output_path = annotation_path(input_path, args.moveset)
    result: dict[str, Any] = {
        "input": input_path,
        "output": str(output_path),
        "moveset": report.moveset,
        "total": report.total,
        "reached_origin": report.reached_origin,
        "with_shorter": report.with_shorter,
        "computed": report.computed,
        "max_distance_to_origin": report.max_distance_to_origin,
    }
    _publish_dataset_artifact(
        args,
        reporter,
        phase="annotate",
        data_path=output_path,
        summary_writer=write_annotation_summary,
        result=result,
    )
    reporter.result_json(result, sort_keys=True)
    return 0


def _dataset_split(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Assign every group to a train/val/test split and publish the split file.

    The assignment is a function of each group's content hash, so re-running this
    after a ``dataset grow`` places the new groups without disturbing any group a
    model has already been evaluated on.
    """
    input_path = _resolve_dataset_path(args.input)
    val, test = args.val_fraction, args.test_fraction
    config = SplitConfig(train=1.0 - val - test, val=val, test=test, salt=args.split_salt)
    report = write_split(input_path, config)
    result: dict[str, Any] = {
        "input": input_path,
        "output": report.path,
        "total": report.total,
        "train": report.train,
        "val": report.val,
        "test": report.test,
    }
    _publish_dataset_artifact(
        args,
        reporter,
        phase="split",
        data_path=Path(report.path),
        summary_writer=None,
        result=result,
    )
    reporter.result_json(result, sort_keys=True)
    return 0


def _dataset_labels(args: argparse.Namespace, reporter: CliReporter) -> int:
    """Precompute the supervised label and instance sidecars for a training config.

    Run this once before ``aczero train`` on a large ball so the run memory-maps ready-made
    sidecars instead of building them at startup. It provisions the dataset the same way
    ``train`` does (refreshing groups/annotations from the bucket and ensuring the split)
    and then builds exactly the sidecars that config's run would build -- same moveset,
    bound, and split -- fanning the per-group move-application across ``--workers`` processes.
    """
    config = TrainingPipelineConfig.from_mapping(_load_config(Path(args.config)))
    if args.workers:
        config = replace(config, workers=args.workers)
    config = _seed_from_default_dataset(config)
    _ensure_training_dataset(config, reporter)
    groups = Path(str(config.dataset_path))
    annotations = Path(str(config.dataset_annotations_path))
    split_file = _split_file_for(config, groups)

    def on_build(message: str, metrics: dict[str, Any]) -> None:
        reporter.progress("sidecar", message, metrics)

    instances = InstanceStore.open(groups, annotations, progress=on_build)
    labels = SupervisedStore.open(
        groups,
        annotations,
        split_file,
        config.moveset,
        config.max_relator_tokens,
        workers=config.workers,
        progress=on_build,
    )
    reporter.result_json(
        {
            "groups": str(groups),
            "labels": str(labels.path),
            "instances": str(instances.path),
            "count": labels.count,
            "actions": labels.actions,
            "moveset": labels.moveset,
        },
        sort_keys=True,
    )
    return 0


def _split_file_for(config: TrainingPipelineConfig, groups: Path) -> Path:
    """The split path a supervised run reads: the configured one, else the one beside groups."""
    return Path(config.dataset_split_path) if config.dataset_split_path else split_path(groups)


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
