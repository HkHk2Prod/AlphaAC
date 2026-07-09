import json
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


def test_scheduler_notebook_training_fills_the_runtime_budget() -> None:
    source = _code_source(_load(SCHEDULER_NOTEBOOK))
    # `training.iterations` is an upper bound only: the wall-clock budget ends the
    # run, and training stops earlier than the watchdog so its final checkpoint
    # upload is not interrupted by the terminate().
    assert "TRAIN_BUDGET_MIN = max(1, JOB_BUDGET_MIN - TRAIN_FLUSH_MARGIN_MIN)" in source
    assert '"--minutes", str(TRAIN_BUDGET_MIN)' in source
    assert '"--minutes", str(JOB_BUDGET_MIN)' in source  # generation fills the whole budget


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
    assert "upload_dataset(dataset_path" not in source
    assert "upload_dataset(path" in source


@pytest.mark.parametrize("name", ("01_generate_dataset.ipynb", "02_annotate_dataset.ipynb"))
def test_fails_fast_when_upload_is_required_but_no_hf_token(name: str) -> None:
    source = _code_source(_load(name))
    # A missing HF_TOKEN would only surface as a failed upload after the whole
    # time budget was spent; raise immediately instead so nothing is wasted.
    assert 'if HF_UPLOAD_ON_FINISH and not os.environ.get("HF_TOKEN"):' in source
    assert "raise RuntimeError(" in source
