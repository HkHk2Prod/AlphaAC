import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "kaggle"
NOTEBOOKS = ("01_generate_dataset.ipynb", "02_annotate_dataset.ipynb", "03_train.ipynb")


def _load(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def _code_source(nb: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_nbformat_v4(name: str) -> None:
    nb = _load(name)
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    assert any(c["cell_type"] == "code" for c in nb["cells"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_compiles(name: str) -> None:
    nb = _load(name)
    for index, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
        source = "".join(cell["source"])
        # The notebooks use plain Python (subprocess pip), no IPython magics.
        assert not source.lstrip().startswith(("!", "%")), f"{name} cell {index} uses a magic"
        compile(source, f"{name}:code[{index}]", "exec")


@pytest.mark.parametrize("name", NOTEBOOKS)
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


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_writes_to_kaggle_working(name: str) -> None:
    source = _code_source(_load(name))
    assert "/kaggle/working" in source


def test_training_notebook_seeds_self_play_from_hf_dataset() -> None:
    source = _code_source(_load("03_train.ipynb"))
    # Pulls the grown dataset from the bucket and hands its path to the config.
    assert "download_dataset" in source and "HF_BUCKET" in source
    assert "HF_DOWNLOAD_ON_START" in source
    assert '"path": DATASET_PATH' in source


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
