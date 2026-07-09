"""KaggleClient: push/status wrapper and status normalization."""

from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from ac_zero.scheduler.kaggle import KaggleClient, KaggleError


def _runner(rc: int, out: str = "", err: str = ""):
    def run(argv: list[str]) -> CompletedProcess[str]:
        return CompletedProcess(argv, rc, out, err)

    return run


@pytest.mark.parametrize(
    ("raw", "expected", "terminal"),
    [
        ('x has status "complete"', "complete", True),  # classic 1.x
        ('x has status "error"', "error", True),
        ('x has status "KernelWorkerStatus.ERROR"', "error", True),  # 2.x enum
        ('x has status "KernelWorkerStatus.COMPLETE"', "complete", True),
        ('x has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "cancelacknowledged", True),
        ('x has status "KernelWorkerStatus.RUNNING"', "running", False),
    ],
)
def test_status_parses_both_cli_formats(raw: str, expected: str, terminal: bool) -> None:
    client = KaggleClient(runner=_runner(0, raw))
    status = client.status("u/x")
    assert status == expected
    assert KaggleClient.is_terminal(status) is terminal


def test_status_none_on_cli_failure() -> None:
    client = KaggleClient(runner=_runner(1, "", "not found"))
    assert client.status("u/x") is None


def test_push_ok_returns_output() -> None:
    client = KaggleClient(runner=_runner(0, "Kernel version 1 successfully pushed."))
    result = client.push("dir")
    assert result.ok and "successfully pushed" in result.output


def test_push_raises_on_nonzero() -> None:
    client = KaggleClient(runner=_runner(1, "", "boom"))
    with pytest.raises(KaggleError, match="failed"):
        client.push("dir")
