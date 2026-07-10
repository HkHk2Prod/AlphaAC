"""Per-iteration alpha updating and evaluation metrics for the navigation reward.

The training pipeline holds a single :class:`AlphaUpdater` for the run. Episodes
are collected in parallel and reassembled in a deterministic order, so alpha is
held constant across a whole iteration's batch (and therefore constant within
every episode) and advanced once, episode by episode in that fixed order, after
the batch is collected. :func:`fold_alpha` performs that advance and yields the
per-episode log rows (item 1 of the reward spec); :func:`navigation_eval_metrics`
summarizes the batch (item 7).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ac_zero.environment.navigation_reward import AlphaUpdater, EpisodeStats
from ac_zero.training.callbacks import CallbackManager
from ac_zero.training.events import LogLevel
from ac_zero.training.navigation_curriculum import CurriculumUpdate, DistanceCurriculum
from ac_zero.training.pipeline_episodes import EpisodeMetrics

# Out-of-range sentinel for the frontier success EMA before any frontier episode
# has seeded it (the real value is always in [0, 1]).
_EMA_UNSET = -1.0


def _episode_stats(episodes: Sequence[EpisodeMetrics]) -> list[EpisodeStats]:
    return [episode.nav for episode in episodes if episode.nav is not None]


def fold_alpha(updater: AlphaUpdater, episodes: Sequence[EpisodeMetrics]) -> list[dict[str, float]]:
    """Advance ``updater`` over each navigation episode, in collection order.

    Returns one row per folded episode carrying the values the spec asks to log
    every episode: the resulting ``alpha`` plus ``progress_rate``/``success`` and
    their EMAs. ``updater`` is left holding the alpha the next iteration runs at.
    """
    rows: list[dict[str, float]] = []
    for stats in _episode_stats(episodes):
        alpha = updater.update(stats)
        rows.append(
            {
                "alpha": alpha,
                "progress_rate": stats.progress_rate,
                "progress_ema": updater.progress_ema,
                "success": 1.0 if stats.success else 0.0,
                "success_ema": updater.success_ema,
            }
        )
    return rows


def navigation_eval_metrics(episodes: Sequence[EpisodeMetrics]) -> dict[str, float]:
    """Average the navigation episodes' distances, lengths, and reward components.

    Returns an empty mapping when no episode carried navigation stats, so a
    non-navigation run contributes nothing.
    """
    stats = _episode_stats(episodes)
    if not stats:
        return {}
    count = float(len(stats))

    def avg(values: list[float]) -> float:
        return sum(values) / count

    return {
        "success_rate": avg([1.0 if s.success else 0.0 for s in stats]),
        "progress_rate": avg([s.progress_rate for s in stats]),
        "average_min_distance_reached": avg([float(s.min_distance_reached) for s in stats]),
        "average_final_distance": avg([float(s.final_distance) for s in stats]),
        "average_episode_length": avg([float(s.length) for s in stats]),
        "average_revisit_count": avg([float(s.revisit_count) for s in stats]),
        "average_destination_reward": avg([s.destination_reward for s in stats]),
        "average_shaping_reward": avg([s.shaping_reward for s in stats]),
        "average_move_fee": avg([s.move_fee for s in stats]),
        "average_revisit_fee": avg([s.revisit_fee for s in stats]),
        "average_total_reward": avg([s.total_reward for s in stats]),
    }


def log_navigation(
    manager: CallbackManager,
    base_event_id: int,
    iteration: int,
    updater: AlphaUpdater,
    episodes: Sequence[EpisodeMetrics],
    level: LogLevel,
) -> None:
    """Advance ``updater`` from the batch and emit the per-episode + aggregate events.

    Per-episode rows carry alpha and the progress/success EMAs (item 1); the
    aggregate event carries the evaluation metrics (item 7). No-op when the batch
    holds no navigation episodes.
    """
    for offset, row in enumerate(fold_alpha(updater, episodes)):
        manager.emit(
            base_event_id + offset,
            "navigation_episode",
            "navigation episode alpha update",
            {"iteration": iteration, **row},
            level=level,
        )
    metrics = navigation_eval_metrics(episodes)
    if metrics:
        manager.emit(
            base_event_id + len(episodes) + 1,
            "navigation",
            "navigation evaluation metrics",
            {
                "iteration": iteration,
                "alpha": updater.alpha,
                "progress_ema": updater.progress_ema,
                "success_ema": updater.success_ema,
                **metrics,
            },
            level=level,
        )


def fold_curriculum(
    curriculum: DistanceCurriculum, episodes: Sequence[EpisodeMetrics], L_max_episode: int
) -> list[CurriculumUpdate]:
    """Advance ``curriculum`` over each navigation episode, in collection order.

    ``L_max_episode`` is the ceiling the batch was sampled under; it is what
    decides each episode's frontier membership, independent of any ``L_max``
    change the fold itself triggers partway through.
    """
    return [
        curriculum.update(
            L=stats.start_distance, success=stats.success, L_max_episode=L_max_episode
        )
        for stats in _episode_stats(episodes)
    ]


def _curriculum_row(update: CurriculumUpdate) -> dict[str, float | int | bool | str]:
    ema = update.frontier_success_ema
    return {
        "L": update.L,
        "L_max": update.L_max,
        "L_max_episode": update.L_max_episode,
        "max_moves": 3 * update.L + 6,
        "frontier_lower": update.frontier_lower,
        "is_frontier_episode": update.is_frontier_episode,
        "frontier_success_ema": _EMA_UNSET if ema is None else ema,
        "frontier_success_count": update.frontier_success_count,
        "episodes_since_lmax_change": update.episodes_since_lmax_change,
        "L_max_changed": update.L_max_changed,
        "L_max_change_direction": update.L_max_change_direction,
    }


def _rate_by_key(keys: Sequence[str], successes: Sequence[float]) -> str:
    """Render ``key:success_rate`` pairs (sorted by key) as a compact string."""
    totals: dict[str, list[float]] = {}
    for key, success in zip(keys, successes, strict=True):
        totals.setdefault(key, []).append(success)
    return "|".join(f"{key}:{round(sum(v) / len(v), 3)}" for key, v in sorted(totals.items()))


def curriculum_aggregate(
    updates: Sequence[CurriculumUpdate], episodes: Sequence[EpisodeMetrics]
) -> dict[str, float | int | bool | str]:
    """Batch-level curriculum metrics: frontier share and success by distance/band."""
    stats = _episode_stats(episodes)
    if not updates or not stats:
        return {}
    count = len(updates)
    successes = [1.0 if s.success else 0.0 for s in stats]
    distances = [u.L for u in updates]
    histogram = Counter(distances)
    return {
        "frontier_episode_fraction": sum(u.is_frontier_episode for u in updates) / count,
        "sampled_distance_mean": sum(distances) / count,
        "sampled_distance_histogram": "|".join(f"{d}:{histogram[d]}" for d in sorted(histogram)),
        "success_rate_by_distance": _rate_by_key([str(d) for d in distances], successes),
        "success_rate_by_frontier_status": _rate_by_key(
            ["frontier" if u.is_frontier_episode else "interior" for u in updates], successes
        ),
    }


def log_curriculum(
    manager: CallbackManager,
    base_event_id: int,
    iteration: int,
    curriculum: DistanceCurriculum,
    episodes: Sequence[EpisodeMetrics],
    L_max_episode: int,
    level: LogLevel,
) -> None:
    """Fold the distance curriculum from the batch and emit per-episode + aggregate events.

    Per-episode rows carry the distance/frontier/``L_max``-change fields (item 8);
    the aggregate event carries the batch's frontier share and success breakdowns.
    No-op when the batch holds no navigation episodes.
    """
    updates = fold_curriculum(curriculum, episodes, L_max_episode)
    for offset, update in enumerate(updates):
        manager.emit(
            base_event_id + offset,
            "curriculum_episode",
            "distance curriculum episode",
            {"iteration": iteration, **_curriculum_row(update)},
            level=level,
        )
    aggregate = curriculum_aggregate(updates, episodes)
    if aggregate:
        manager.emit(
            base_event_id + len(episodes) + 1,
            "curriculum",
            "distance curriculum metrics",
            {"iteration": iteration, "L_max": curriculum.L_max, **aggregate},
            level=level,
        )
    _log_length_cap_changes(manager, base_event_id + len(episodes) + 2, iteration, updates)


def _log_length_cap_changes(
    manager: CallbackManager,
    base_event_id: int,
    iteration: int,
    updates: Sequence[CurriculumUpdate],
) -> None:
    """Emit an INFO event for each episode that moved the curriculum's ``L_max``.

    Always INFO (never throttled by the progress level) so a length-cap change --
    a rare, significant milestone -- reaches the console on every verbosity level
    the moment it happens, not only on a progress-report iteration.
    """
    for offset, update in enumerate(u for u in updates if u.L_max_changed):
        manager.emit(
            base_event_id + offset,
            "length_cap",
            f"distance curriculum length cap {update.L_max_change_direction}d",
            {
                "iteration": iteration,
                "L_max": update.L_max,
                "direction": update.L_max_change_direction,
                "max_moves": 3 * update.L_max + 6,
            },
            level=LogLevel.INFO,
        )
