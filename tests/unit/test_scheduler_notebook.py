"""Notebook-side helpers: runtime-config loading and the run reporter.

The reporter publishes through a :class:`StateBackend`, so these tests inject a
:class:`MemoryStateBackend` via the backend factory -- no real ``huggingface_hub``
install or network is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ac_zero.scheduler.backend as backend
from ac_zero.scheduler import notebook as nb
from ac_zero.scheduler.backend import MemoryStateBackend
from ac_zero.scheduler.store import QUEUE_PATH


def test_load_runtime_config_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps({"run_id": "r1", "mode": "generation"}), encoding="utf-8")
    cfg = nb.load_runtime_config(str(path))
    assert cfg["run_id"] == "r1"


def test_load_runtime_config_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="runtime config not found"):
        nb.load_runtime_config(str(tmp_path / "nope.json"))


def test_load_runtime_config_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps({"mode": "generation"}), encoding="utf-8")  # no run_id
    with pytest.raises(RuntimeError, match="malformed runtime config"):
        nb.load_runtime_config(str(path))


def test_login_from_secret_dataset_rejects_bad_token(tmp_path: Path) -> None:
    token_file = tmp_path / "hf_token.txt"
    token_file.write_text("not-a-token\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not look like a Hugging Face token"):
        nb.login_from_secret_dataset(str(token_file))


def test_login_from_secret_dataset_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing Hugging Face token"):
        nb.login_from_secret_dataset(str(tmp_path / "absent.txt"))


def test_locate_token_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "hf_token.txt"
    explicit.write_text("hf_explicit\n", encoding="utf-8")
    assert nb._locate_token(str(explicit)) == explicit


def test_locate_token_searches_mount_when_path_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Emulate the kaggle 2.x mount: /kaggle/input/datasets/<owner>/<slug>/hf_token.txt
    nested = tmp_path / "datasets" / "hkhk2prod" / "runtime-secrets"
    nested.mkdir(parents=True)
    (nested / "hf_token.txt").write_text("hf_found\n", encoding="utf-8")
    monkeypatch.setattr(nb, "KAGGLE_INPUT_ROOT", str(tmp_path))
    found = nb._locate_token(str(tmp_path / "runtime-secrets" / "hf_token.txt"))
    assert found is not None and found.read_text().strip() == "hf_found"


def test_locate_token_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(nb, "KAGGLE_INPUT_ROOT", str(tmp_path))
    assert nb._locate_token(str(tmp_path / "nope.txt")) is None


def _reporter(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, str] | None = None
) -> tuple[nb.RunReporter, MemoryStateBackend]:
    mem = MemoryStateBackend(files or {})
    monkeypatch.setattr(backend, "make_state_backend", lambda *a, **k: mem)
    reporter = nb.RunReporter("u/state", run_id="r1", task_id="gen", token="hf_x")
    return reporter, mem


def test_reporter_started_and_finished_publish_run_files(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, mem = _reporter(monkeypatch)
    reporter.started()
    reporter.finished(status="failed", error="boom")
    run_file = mem.read_text("queue/runs/r1.json")
    latest = mem.read_text("queue/runs/latest.json")
    assert run_file is not None and latest is not None
    record = json.loads(latest)
    assert record["status"] == "failed" and record["error"] == "boom"
    # The token must never leak into a published payload.
    assert "hf_x" not in run_file and "hf_x" not in latest


def test_should_stop_true_when_task_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, _ = _reporter(monkeypatch, {QUEUE_PATH: "tasks:\n  - id: gen\n    active: false\n"})
    assert reporter.should_stop() is True


def test_should_stop_true_on_stop_after_current_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "tasks:\n  - id: gen\n    active: true\n    stop_after_current_iteration: true\n"
    reporter, _ = _reporter(monkeypatch, {QUEUE_PATH: body})
    assert reporter.should_stop() is True


def test_should_stop_false_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, _ = _reporter(monkeypatch, {QUEUE_PATH: "tasks:\n  - id: gen\n    active: true\n"})
    assert reporter.should_stop() is False


def test_should_stop_false_on_missing_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, _ = _reporter(monkeypatch)  # empty backend -> no queue.yaml
    assert reporter.should_stop() is False


def test_should_stop_false_on_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, mem = _reporter(monkeypatch)

    def _boom(_path: str) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(mem, "read_text", _boom)
    assert reporter.should_stop() is False
