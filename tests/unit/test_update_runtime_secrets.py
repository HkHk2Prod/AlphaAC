"""The Kaggle runtime-secrets updater's create-vs-version fallback logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_kaggle_runtime_secrets.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("update_kaggle_runtime_secrets", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proc(rc: int, out: str = "") -> CompletedProcess[str]:
    return CompletedProcess(["kaggle"], rc, out, "")


def test_version_with_prune_wins_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load()
    calls: list[list[str]] = []

    def _run(argv: list[str]) -> CompletedProcess[str]:
        calls.append(argv)
        return _proc(0)

    monkeypatch.setattr(mod, "_run", _run)
    mod._publish(tmp_path, "u/runtime-secrets", "runtime-secrets")
    # Only the prune-version call was made -- no plain-version, no create.
    assert len(calls) == 1
    assert "--delete-old-versions" in calls[0]
    assert (tmp_path / "dataset-metadata.json").exists()


def test_falls_back_to_create_when_dataset_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load()
    seen: list[str] = []

    def _run(argv: list[str]) -> CompletedProcess[str]:
        verb = argv[1] + "/" + argv[2]  # e.g. "datasets/version"
        seen.append(verb)
        # Both version attempts fail (no dataset yet); create succeeds.
        if argv[2] == "create":
            return _proc(0)
        return _proc(1, "404 - Not Found")

    monkeypatch.setattr(mod, "_run", _run)
    mod._publish(tmp_path, "u/runtime-secrets", "runtime-secrets")
    assert seen == ["datasets/version", "datasets/version", "datasets/create"]


def test_surfaces_all_output_when_everything_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load()

    def _run(argv: list[str]) -> CompletedProcess[str]:
        return _proc(1, f"boom-{argv[2]}")

    monkeypatch.setattr(mod, "_run", _run)
    with pytest.raises(SystemExit) as excinfo:
        mod._publish(tmp_path, "u/runtime-secrets", "runtime-secrets")
    message = str(excinfo.value)
    assert "could not update or create" in message
    assert "boom-version" in message and "boom-create" in message
