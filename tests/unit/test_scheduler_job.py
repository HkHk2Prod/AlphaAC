"""Tests for the scheduled-job runner: exit classification and the flush wait.

The bug these pin down: a job the watchdog stops is terminated, so it exits with
-SIGTERM whatever it did -- read literally, every budget-capped run reports as failed
and the scheduler records an error for a run that finished normally.
"""

import subprocess
import threading

from ac_zero.scheduler.job import FLUSH_MARKER, JobOutcome, JobRunner


class FakeProcess:
    """A subprocess whose stdout and exit are scripted by the test.

    ``lines`` are handed out one per ``readline``; the process stays alive until they
    run out (then it exits with ``returncode``), unless the test terminates it first.
    """

    def __init__(self, lines=(), returncode=0, exits_on_terminate=True):
        self._lines = list(lines)
        self._returncode = returncode
        self._exits_on_terminate = exits_on_terminate
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = self

    # -- stdout surface -------------------------------------------------
    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._exit(self._returncode)
        return ""

    def __iter__(self):
        while self._lines:
            yield self._lines.pop(0)

    # -- process surface ------------------------------------------------
    def _exit(self, rc):
        if self.returncode is None:
            self.returncode = rc

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("cmd", timeout or 0)
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self._exits_on_terminate:
            self._exit(-15)

    def kill(self):
        self.killed = True
        self._exit(-9)


def make_runner(proc, *, stop=False, deadline=False):
    """A runner over ``proc``, with the watchdog's events already in their end state."""
    stop_event, deadline_hit = threading.Event(), threading.Event()
    if stop:
        stop_event.set()
    if deadline:
        deadline_hit.set()
        stop_event.set()
    emitted = []
    return (
        JobRunner(
            ["aczero", "train"],
            stop_event=stop_event,
            deadline_hit=deadline_hit,
            popen=lambda _cmd: proc,
            sleep=lambda _s: None,
            monotonic=lambda: 0.0,
            emit=emitted.append,
        ),
        emitted,
    )


def test_job_that_exits_cleanly_is_finished():
    runner, emitted = make_runner(FakeProcess(lines=["training\n"], returncode=0))
    assert runner.run() == JobOutcome("finished")
    assert "training" in emitted


def test_job_that_exits_nonzero_is_failed():
    runner, _ = make_runner(FakeProcess(returncode=3))
    outcome = runner.run()
    assert outcome.status == "failed"
    assert outcome.error == "job exited with code 3"


def test_deadline_termination_is_finished_not_failed():
    """The -SIGTERM we sent is ours, not the job's: a deadline stop is a normal end.

    This is the regression: the old cell overwrote the deadline's "finished" with
    "failed" because rc was -15, so Kaggle marked the whole notebook version failed.
    """
    proc = FakeProcess()
    runner, _ = make_runner(proc, deadline=True)
    outcome = runner.run()
    assert proc.terminated
    assert outcome.status == "finished"
    assert outcome.error is None


def test_operator_stop_termination_is_stopped_not_failed():
    proc = FakeProcess()
    runner, _ = make_runner(proc, stop=True)
    outcome = runner.run()
    assert proc.terminated
    assert outcome.status == "stopped"
    assert outcome.error is None


def test_job_flushing_at_the_deadline_is_waited_for_not_terminated():
    """A job that announced its flush is writing the checkpoint -- let it finish."""
    proc = FakeProcess(
        lines=[f"budget: {FLUSH_MARKER}; stopping at iteration boundary\n", "pushed bundle\n"],
        returncode=0,
    )
    stop_event, deadline_hit = threading.Event(), threading.Event()
    emitted = []

    # The stop lands only after the job has announced the flush, as on a real run.
    def relay_then_stop(line):
        emitted.append(line)
        if FLUSH_MARKER in line:
            deadline_hit.set()
            stop_event.set()

    runner = JobRunner(
        ["aczero", "train"],
        stop_event=stop_event,
        deadline_hit=deadline_hit,
        popen=lambda _cmd: proc,
        sleep=lambda _s: None,
        monotonic=lambda: 0.0,
        emit=relay_then_stop,
    )
    outcome = runner.run()
    assert not proc.terminated, "terminating a flushing job throws its checkpoint away"
    assert "pushed bundle" in emitted
    assert outcome.status == "finished"


def test_flush_that_overruns_its_grace_window_is_terminated():
    proc = FakeProcess(lines=[f"{FLUSH_MARKER}\n"], exits_on_terminate=True)
    stop_event, deadline_hit = threading.Event(), threading.Event()
    emitted = []

    def relay_then_stop(line):
        emitted.append(line)
        if FLUSH_MARKER in line:
            deadline_hit.set()
            stop_event.set()

    # Clock jumps past the grace window on the first check inside the flush wait.
    ticks = iter([0.0, 10_000.0, 10_000.0])
    runner = JobRunner(
        ["aczero", "train"],
        stop_event=stop_event,
        deadline_hit=deadline_hit,
        flush_grace_s=60,
        popen=lambda _cmd: proc,
        sleep=lambda _s: None,
        monotonic=lambda: next(ticks),
        emit=relay_then_stop,
    )
    outcome = runner.run()
    assert proc.terminated
    assert outcome.status == "finished"  # deadline stop, still not a failure


def test_job_killed_when_it_ignores_terminate():
    proc = FakeProcess(exits_on_terminate=False)
    runner, _ = make_runner(proc, deadline=True)
    runner.run()
    assert proc.terminated and proc.killed


def test_popen_failure_is_reported_as_failed():
    stop_event, deadline_hit = threading.Event(), threading.Event()

    def boom(_cmd):
        raise FileNotFoundError("aczero")

    runner = JobRunner(
        ["aczero", "train"], stop_event=stop_event, deadline_hit=deadline_hit, popen=boom
    )
    outcome = runner.run()
    assert outcome.status == "failed"
    assert "FileNotFoundError" in (outcome.error or "")
