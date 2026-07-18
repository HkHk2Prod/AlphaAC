"""Per-iteration alpha updating and evaluation metrics for the navigation reward.

The training pipeline holds a single :class:`AlphaUpdater` for the run. Episodes
are collected in parallel and reassembled in a deterministic order, so alpha is
held constant across a whole iteration's batch (and therefore constant within
every episode) and advanced once, episode by episode in that fixed order, after
the batch is collected. :func:`fold_alpha` performs that advance and yields the
per-episode log rows (item 1 of the reward spec); :func:`navigation_eval_metrics`
summarizes the batch (item 7).

Alpha is the run's one adaptive difficulty knob: it both scales the shaping term
and, through it, how much of the path to the destination the agent is effectively
being credited for. There is no second sampling-side ceiling to keep in step with
it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ac_zero.environment.navigation_reward import AlphaUpdater, EpisodeStats
from ac_zero.training.logging.callbacks import CallbackManager
from ac_zero.training.logging.events import LogLevel
from ac_zero.training.pipeline.pipeline_episodes import EpisodeMetrics


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
