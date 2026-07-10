"""Artifacts a training run writes once its iteration loop has ended.

Kept out of :mod:`ac_zero.training.pipeline` so that module stays about the
training loop. Two things happen here: the run solves a small fixture with the
greedy solver and checks the resulting certificate verifies (a self-check that
the environment and verifier still agree), and it renders the progress plots,
which degrade to a warning when matplotlib is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ac_zero.agents.greedy import GreedySolver
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.environment.env import ACEnvironment
from ac_zero.training.callbacks import CallbackManager
from ac_zero.training.events import LogLevel
from ac_zero.training.pipeline_config import TrainingPipelineConfig
from ac_zero.training.pipeline_episodes import build_env_config, moves_for_distance
from ac_zero.training.plots import PlotsUnavailable, render_training_plots


def write_fixture_certificate(
    config: TrainingPipelineConfig, seed: int, certificate_path: Path
) -> bool:
    """Solve a small fixture, write its certificate, and report whether it verifies."""
    depth = min(config.scramble_depth, 2)
    fixture = generate_solvable(config.rank, depth, seed)
    # The fixture is a depth-``depth`` scramble, so its distance to the trivial
    # group is at most ``depth``; the ``3 * L + 6`` horizon at that bound gives the
    # greedy self-check ample room without a pathological deep search.
    solve_env = ACEnvironment(
        fixture.presentation, build_env_config(config, None, moves_for_distance(depth))
    )
    result = GreedySolver().solve(
        solve_env, certificate_path=certificate_path, experiment_id="training", seed=seed
    )
    return bool(result.success and CertificateVerifier().verify_path(certificate_path).ok)


def render_plots(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    manager: CallbackManager,
    event_id: int,
) -> tuple[str, ...]:
    """Render training-progress plots, reporting the outcome through ``manager``.

    Returns the written PNG paths. If matplotlib is not installed the run still
    succeeds — a warning is logged pointing at the always-available ASCII graphs
    and an empty tuple is returned.
    """
    try:
        paths = render_training_plots(rows, output_dir)
    except PlotsUnavailable:
        manager.emit(
            event_id,
            "plots",
            "matplotlib not installed; skipping image plots (ASCII graphs still written)",
            {"matplotlib": False},
            level=LogLevel.WARNING,
        )
        return ()
    if paths:
        manager.emit(
            event_id,
            "plots",
            "rendered training-progress plots",
            {"count": len(paths), "directory": str(output_dir)},
        )
    return tuple(str(path) for path in paths)
