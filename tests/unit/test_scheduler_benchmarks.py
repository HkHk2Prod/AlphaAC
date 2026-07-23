from __future__ import annotations

import json
from typing import Any

import pytest

from ac_zero.scheduler import benchmarks as benchmarks_module
from ac_zero.scheduler.backend import MemoryStateBackend
from ac_zero.scheduler.benchmarks import (
    BENCHMARK_QUEUE_PATH,
    MAX_DISPATCHED_HISTORY,
    BenchmarkQueue,
    PendingEvaluation,
    scan_for_ready_checkpoints,
)
from ac_zero.scheduler.models import Queue, Task

_TRAINING_CONFIG: dict[str, Any] = {
    "rank": 2,
    "moveset": "strict-ac",
    "agent": "ppo",
    "model": "residual_mlp",
    "training": {"reward_mode": "navigation"},
}


def _training_task(task_id: str = "train-a", **config: Any) -> Task:
    return Task(
        id=task_id,
        mode="training",
        notebook_slug="o/s",
        notebook_dir="notebooks/kaggle",
        config={**_TRAINING_CONFIG, **config},
    )


def _queue(*tasks: Task) -> Queue:
    return Queue(tasks=list(tasks))


def _index(metric: float | None, run_id: str = "run-1", fmt: int = 1) -> dict[str, Any]:
    best = {"metric": metric, "run_id": run_id, "format_version": fmt}
    return {"name": "n", "best": best, "runs": []}


def _scan(tasks: Queue, queue: BenchmarkQueue, **kwargs: Any) -> list[PendingEvaluation]:
    return scan_for_ready_checkpoints(
        tasks, queue, bucket="b/c", threshold=0.30, log=lambda _: None, **kwargs
    )


def test_a_missing_document_loads_as_an_empty_queue() -> None:
    queue = BenchmarkQueue.load(MemoryStateBackend())
    assert queue.pending == []
    assert queue.dispatched == []


def test_a_malformed_document_loads_as_an_empty_queue() -> None:
    backend = MemoryStateBackend({BENCHMARK_QUEUE_PATH: "{not json"})
    assert BenchmarkQueue.load(backend).pending == []


def test_the_queue_round_trips_through_its_document() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    restored = BenchmarkQueue.load(MemoryStateBackend({BENCHMARK_QUEUE_PATH: queue.to_json()}))
    assert [e.key for e in restored.pending] == [("model-a", "run-1")]
    assert restored.pending[0].metric == 0.5


def test_enqueue_stamps_the_time_it_landed() -> None:
    queue = BenchmarkQueue()
    assert queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    assert queue.pending[0].enqueued_at.endswith("Z")


def test_the_same_run_is_never_queued_twice() -> None:
    queue = BenchmarkQueue()
    assert queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    assert not queue.enqueue(PendingEvaluation("model-a", "run-1", 0.9))
    assert len(queue.pending) == 1


def test_a_dispatched_run_is_not_re_queued() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    queue.take()
    assert not queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    assert queue.pending == []


def test_a_later_run_of_the_same_model_is_a_separate_entry() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    assert queue.enqueue(PendingEvaluation("model-a", "run-2", 0.6))
    assert len(queue.pending) == 2


def test_take_pops_the_highest_metric_first() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.4))
    queue.enqueue(PendingEvaluation("model-b", "run-2", 0.9))
    queue.enqueue(PendingEvaluation("model-c", "run-3", 0.6))
    assert [queue.take().checkpoint_name for _ in range(3)] == [  # type: ignore[union-attr]
        "model-b",
        "model-c",
        "model-a",
    ]


def test_take_on_an_empty_queue_returns_none() -> None:
    assert BenchmarkQueue().take() is None


def test_dispatched_history_is_trimmed() -> None:
    queue = BenchmarkQueue()
    for index in range(MAX_DISPATCHED_HISTORY + 10):
        queue.enqueue(PendingEvaluation("model-a", f"run-{index}", 0.5))
        queue.take()
    assert len(queue.dispatched) == MAX_DISPATCHED_HISTORY


def test_the_gate_queues_a_checkpoint_that_clears_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(0.42))
    queue = BenchmarkQueue()
    added = scan_for_ready_checkpoints(
        _queue(_training_task()), queue, bucket="b/c", threshold=0.30, log=lambda _: None
    )
    assert len(added) == 1
    assert added[0].metric == 0.42
    assert len(queue.pending) == 1


def test_the_gate_ignores_a_checkpoint_below_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(0.10))
    queue = BenchmarkQueue()
    assert (
        scan_for_ready_checkpoints(
            _queue(_training_task()), queue, bucket="b/c", threshold=0.30, log=lambda _: None
        )
        == []
    )
    assert queue.pending == []


def test_the_gate_is_idempotent_across_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(0.9))
    queue = BenchmarkQueue()
    tasks = _queue(_training_task())
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=lambda _: None)
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=lambda _: None)
    assert len(queue.pending) == 1


def test_the_gate_skips_non_training_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(name: str, *, bucket: str) -> dict[str, Any]:
        raise AssertionError("a non-training task must not be inspected")

    monkeypatch.setattr(benchmarks_module, "_read_index", explode)
    ball = Task(id="ball", mode="ball", notebook_slug="o/s", notebook_dir="d")
    assert (
        scan_for_ready_checkpoints(
            _queue(ball), BenchmarkQueue(), bucket="b/c", threshold=0.3, log=lambda _: None
        )
        == []
    )


def test_a_bucket_error_does_not_fail_the_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str, *, bucket: str) -> dict[str, Any]:
        raise RuntimeError("bucket unavailable")

    monkeypatch.setattr(benchmarks_module, "_read_index", boom)
    logged: list[str] = []
    assert (
        scan_for_ready_checkpoints(
            _queue(_training_task()),
            BenchmarkQueue(),
            bucket="b/c",
            threshold=0.3,
            log=logged.append,
        )
        == []
    )
    assert any("bucket unavailable" in line for line in logged)


def test_a_checkpoint_without_a_published_best_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: None)
    assert (
        scan_for_ready_checkpoints(
            _queue(_training_task()), BenchmarkQueue(), bucket="b/c", threshold=0.3, log=print
        )
        == []
    )


def test_an_index_with_a_null_metric_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(None))
    assert (
        scan_for_ready_checkpoints(
            _queue(_training_task()), BenchmarkQueue(), bucket="b/c", threshold=0.3, log=print
        )
        == []
    )


def test_an_explicit_checkpoint_name_in_the_task_config_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def record(name: str, *, bucket: str) -> dict[str, Any]:
        seen.append(name)
        return _index(0.9)

    monkeypatch.setattr(benchmarks_module, "_read_index", record)
    scan_for_ready_checkpoints(
        _queue(_training_task(checkpoint_name="pinned-name")),
        BenchmarkQueue(),
        bucket="b/c",
        threshold=0.3,
        log=lambda _: None,
    )
    assert seen == ["pinned-name"]


def test_a_derived_name_matches_what_the_trainer_would_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ac_zero.training.checkpointing.checkpoint_name import derive_checkpoint_name
    from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig

    expected = derive_checkpoint_name(TrainingPipelineConfig.from_mapping(_TRAINING_CONFIG))
    seen: list[str] = []
    monkeypatch.setattr(
        benchmarks_module,
        "_read_index",
        lambda name, *, bucket: (seen.append(name), _index(0.9))[1],
    )
    scan_for_ready_checkpoints(
        _queue(_training_task()), BenchmarkQueue(), bucket="b/c", threshold=0.3, log=lambda _: None
    )
    assert seen == [expected]


def test_the_document_is_valid_json_with_both_lists() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    queue.take()
    payload = json.loads(queue.to_json())
    assert payload["pending"] == []
    assert payload["dispatched"][0]["checkpoint_name"] == "model-a"


def test_enqueueing_moves_the_models_ladder_rung() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5), format_version=2)
    rung = queue.ladder["model-a"]
    assert (rung.run_id, rung.metric, rung.format_version) == ("run-1", 0.5, 2)
    assert rung.at.endswith("Z")


def test_a_rejected_enqueue_leaves_the_rung_alone() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5))
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.9))
    assert queue.ladder["model-a"].metric == 0.5


def test_the_ladder_round_trips_through_the_document() -> None:
    queue = BenchmarkQueue()
    queue.enqueue(PendingEvaluation("model-a", "run-1", 0.5), format_version=2)
    restored = BenchmarkQueue.load(MemoryStateBackend({BENCHMARK_QUEUE_PATH: queue.to_json()}))
    assert restored.ladder["model-a"].metric == 0.5
    assert restored.ladder["model-a"].format_version == 2


def test_a_document_with_a_malformed_ladder_loads_without_one() -> None:
    raw = json.dumps({"pending": [], "dispatched": [], "ladder": ["not", "a", "map"]})
    assert BenchmarkQueue.load(MemoryStateBackend({BENCHMARK_QUEUE_PATH: raw})).ladder == {}


def test_the_gate_records_a_rung_for_what_it_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(0.42))
    queue = BenchmarkQueue()
    added = _scan(_queue(_training_task()), queue)
    assert queue.ladder[added[0].checkpoint_name].metric == 0.42


def test_a_later_run_that_has_not_cleared_the_rung_is_not_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexes = iter([_index(0.35, "run-1"), _index(0.49, "run-2")])
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: next(indexes))
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    assert len(_scan(tasks, queue)) == 1
    assert _scan(tasks, queue) == []  # 0.49 is short of the 0.5125 rung


def test_a_later_run_that_clears_the_rung_is_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    indexes = iter([_index(0.35, "run-1"), _index(0.52, "run-2")])
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: next(indexes))
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    _scan(tasks, queue)
    added = _scan(tasks, queue)
    assert [e.metric for e in added] == [0.52]
    assert queue.ladder[added[0].checkpoint_name].run_id == "run-2"


def test_a_format_bump_re_evaluates_a_model_below_its_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexes = iter([_index(0.80, "run-1", fmt=2), _index(0.40, "run-2", fmt=3)])
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: next(indexes))
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    _scan(tasks, queue)
    assert [e.metric for e in _scan(tasks, queue)] == [0.40]


def test_a_held_back_model_is_logged_with_its_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    indexes = iter([_index(0.35, "run-1"), _index(0.49, "run-2")])
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: next(indexes))
    logged: list[str] = []
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=lambda _: None)
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=logged.append)
    assert any("held back" in line and "0.512" in line for line in logged)


def test_a_run_still_holding_its_own_rung_is_logged_about_every_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: _index(0.42))
    logged: list[str] = []
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=lambda _: None)
    scan_for_ready_checkpoints(tasks, queue, bucket="b/c", threshold=0.3, log=logged.append)
    assert logged == []


def test_a_wider_error_reduction_holds_a_model_back_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indexes = iter([_index(0.35, "run-1"), _index(0.60, "run-2")])
    monkeypatch.setattr(benchmarks_module, "_read_index", lambda name, *, bucket: next(indexes))
    queue, tasks = BenchmarkQueue(), _queue(_training_task())
    _scan(tasks, queue, error_reduction=0.5)
    assert _scan(tasks, queue, error_reduction=0.5) == []
