import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "kaggle"
# The hand-run notebooks, each of which owns its own wall-clock deadline.
NOTEBOOKS = ("01_generate_dataset.ipynb", "02_annotate_dataset.ipynb", "03_train.ipynb")
# The scheduler launches this one instead: it shells out to the `aczero` CLI and
# takes its deadline from the runtime config, so it shares only the basic checks.
SCHEDULER_NOTEBOOK = "scheduler_runner.ipynb"
ALL_NOTEBOOKS = (*NOTEBOOKS, SCHEDULER_NOTEBOOK)


def _load(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def _code_source(nb: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_notebook_is_valid_nbformat_v4(name: str) -> None:
    nb = _load(name)
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert any(c["cell_type"] == "code" for c in nb["cells"])


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_every_code_cell_compiles(name: str) -> None:
    nb = _load(name)
    for index, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        source = "".join(cell["source"])
        # The notebooks use plain Python (subprocess pip), no IPython magics.
        assert not source.lstrip().startswith(("!", "%")), f"{name} cell {index} uses a magic"
        compile(source, f"{name}:code[{index}]", "exec")


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_installs_from_github(name: str) -> None:
    source = _code_source(_load(name))
    assert "REPO_URL" in source and "REPO_BRANCH" in source
    assert "git+" in source and "pip" in source


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_respects_a_time_budget(name: str) -> None:
    source = _code_source(_load(name))
    assert "TIME_BUDGET_HOURS" in source
    assert "SAFETY_MARGIN_MIN" in source
    # Each notebook stops cleanly before the Kaggle kill.
    assert "DeadlineReached" in source


@pytest.mark.parametrize("name", ALL_NOTEBOOKS)
def test_writes_to_kaggle_working(name: str) -> None:
    source = _code_source(_load(name))
    assert "/kaggle/working" in source


def test_training_notebook_seeds_self_play_from_hf_dataset() -> None:
    source = _code_source(_load("03_train.ipynb"))
    # Pulls the grown dataset from the bucket and hands its path to the config.
    assert "download_dataset" in source and "HF_BUCKET" in source
    assert "HF_DOWNLOAD_ON_START" in source
    assert '"path": DATASET_PATH' in source


def test_training_notebook_warm_starts_and_uploads_checkpoints() -> None:
    source = _code_source(_load("03_train.ipynb"))
    # Resolves the checkpoint name, warm-starts from the best model on HF, and
    # pushes best model + progress plots via the periodic uploader.
    assert "derive_checkpoint_name" in source
    assert "download_best_checkpoint" in source and "WARM_START" in source
    assert "PeriodicCheckpointUploader" in source
    assert "HF_UPLOAD_EVERY_HOURS" in source
    # The summary shows both this run's plots and the all-runs aggregate.
    assert "plots/all_runs/" in source


def test_scheduler_notebook_jobs_stop_in_time_to_flush_their_output() -> None:
    """Every long job stops itself before the watchdog, with time to write and push.

    A terminated job is killed outright -- nothing in ac_zero handles SIGTERM -- so what
    it had not yet pushed dies with the container. Each job therefore runs on its own
    budget: what is left of the session *now* (so the bucket pull it just paid for comes
    out of its working time, not its margin), less the minutes it needs to flush.
    """
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    assert "remaining = (DEADLINE - time.monotonic()) / 60" in source
    assert "return max(1, int(remaining - flush_margin_min))" in source
    # No job is handed the watchdog's own budget, which would leave it nothing to flush in.
    assert '"--minutes", str(JOB_BUDGET_MIN)' not in source
    assert '"--minutes", str(budget_min(TRAIN_FLUSH_MARGIN_MIN))' in source
    assert '"--minutes", str(budget_min(BALL_FLUSH_MARGIN_MIN))' in source


def test_scheduler_notebook_train_margin_covers_an_iteration_as_well_as_the_flush() -> None:
    """Training's margin must absorb the overshoot before the flush, not just the flush.

    The budget is only noticed at an iteration boundary, so the iteration in flight when
    it expires runs to completion first. At 10 min the margin was smaller than a single
    PPO iteration: the overshoot ate it and the checkpoint write was terminated.
    """
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    match = re.search(r"^TRAIN_FLUSH_MARGIN_MIN = (\d+)", source, re.MULTILINE)
    assert match, "scheduler notebook no longer sets TRAIN_FLUSH_MARGIN_MIN"
    assert int(match.group(1)) >= 25


def test_scheduler_notebook_delegates_the_job_run_to_the_tested_runner() -> None:
    """The exit classification lives in JobRunner, where it is unit-tested.

    Inline in the cell it was untestable, and it silently reported every deadline-stopped
    run as failed (the job's -SIGTERM read as a job error).
    """
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    assert "from ac_zero.scheduler.job import JobRunner" in source
    assert "JobRunner(COMMAND, stop_event=stop_event, deadline_hit=deadline_hit).run()" in source
    # The cell must not second-guess the runner by re-reading the return code itself.
    assert "proc.terminate()" not in source
    assert "job exited with code" not in source


def test_scheduler_notebook_ball_and_training_push_on_a_timed_cadence() -> None:
    """The push is the only durable output, so both jobs make one every few hours."""
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    # The ball rewrites both documents and pushes them; training pushes its bundle.
    assert '"--checkpoint-hours", str(_opt("checkpoint_hours", 4))' in source
    assert '"--upload-every-hours", str(_opt("upload_every_hours", 4))' in source


def test_scheduler_notebook_training_syncs_checkpoints_with_the_bucket() -> None:
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    # Each scheduled training run warm-starts from the lineage's best model on the
    # bucket and pushes its own bundle back, so consecutive runs compound.
    assert '"--download-checkpoint"' in source
    assert '"--upload-checkpoints"' in source
    assert '"--checkpoint-bucket", _bucket()' in source


def test_annotate_notebook_defaults_to_universal_and_strict_ac() -> None:
    source = _code_source(_load("02_annotate_dataset.ipynb"))
    assert 'ANNOTATE_MOVESETS = ["universal", "strict-ac"]' in source


def test_annotate_notebook_warm_starts_from_existing_annotations() -> None:
    source = _code_source(_load("02_annotate_dataset.ipynb"))
    # Pulls any existing annotation file per move set before annotate() runs, so
    # a resumed pass only recomputes groups still unresolved.
    assert "annotation_path" in source
    assert "download_dataset" in source and "missing_ok=True" in source


def test_annotate_notebook_never_uploads_the_group_dataset() -> None:
    source = _code_source(_load("02_annotate_dataset.ipynb"))
    # This notebook runs alongside 01_generate_dataset and must only ever read
    # the group dataset -- it publishes annotation files, never the dataset.
    assert "publish_to_bucket(dataset_path" not in source
    assert "upload_dataset(dataset_path" not in source
    # Its publish step and periodic safeguard only ever push annotation paths.
    assert "for apath in annotation_paths" in source
    assert "publish_to_bucket(" in source


def test_generate_notebook_writes_and_publishes_a_summary() -> None:
    source = _code_source(_load("01_generate_dataset.ipynb"))
    # Uses the shared summary + publish modules (not an inline copy); publish_to_bucket
    # writes the report and pushes it into the bucket's datasets_summaries/ folder.
    assert "write_dataset_summary" in source
    assert "publish_to_bucket(" in source


def test_annotate_notebook_writes_and_publishes_a_summary() -> None:
    source = _code_source(_load("02_annotate_dataset.ipynb"))
    # One summary per annotation file, published via the shared publish_to_bucket.
    assert "write_annotation_summary" in source
    assert "publish_to_bucket(" in source


@pytest.mark.parametrize("name", ("01_generate_dataset.ipynb", "02_annotate_dataset.ipynb"))
def test_fails_fast_when_upload_is_required_but_no_hf_token(name: str) -> None:
    source = _code_source(_load(name))
    # A missing HF_TOKEN would only surface as a failed upload after the whole
    # time budget was spent; raise immediately instead so nothing is wasted.
    assert 'if HF_UPLOAD_ON_FINISH and not os.environ.get("HF_TOKEN"):' in source
    assert "raise RuntimeError(" in source
