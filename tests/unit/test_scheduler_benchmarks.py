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


def _index(metric: float | None, run_id: str = "run-1") -> dict[str, Any]:
    return {"name": "n", "best": {"metric": metric, "run_id": run_id}, "runs": []}


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
