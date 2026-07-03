"""Tests for anchoring bare dataset filenames under ``data/generated``.

The dataset subcommands share an ``--input``/``--output`` pair. A bare filename
(no directory component) should resolve under ``DATASET_DIR`` so that
``aczero dataset download --input train_rank2.json`` lands in
``data/generated/`` rather than the current working directory, and the whole
download → annotate → upload chain agrees on that same path.
"""

from __future__ import annotations

import argparse

import pytest

from ac_zero import cli
from ac_zero.system.reporting import CliReporter


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("train_rank2.json", "data/generated/train_rank2.json"),
        ("data/generated/train_rank2.json", "data/generated/train_rank2.json"),
        ("other/dir/set.json", "other/dir/set.json"),
        ("/abs/set.json", "/abs/set.json"),
        ("", ""),  # unset --output default is left untouched
    ],
)
def test_resolve_dataset_path(value: str, expected: str) -> None:
    assert cli._resolve_dataset_path(value) == expected


def test_download_anchors_bare_input(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_download(local, *, remote_name=None, bucket):  # type: ignore[no-untyped-def]
        seen["local"] = local
        seen["remote_name"] = remote_name
        return local

    monkeypatch.setattr(cli, "download_dataset", fake_download)
    args = argparse.Namespace(input="train_rank2.json", output="", bucket="", remote_name="")
    reporter = CliReporter("dataset")
    try:
        assert cli._dataset_download(args, reporter) == 0
    finally:
        reporter.close()

    assert seen["local"] == "data/generated/train_rank2.json"
    # A bare input keeps its basename as the default remote name.
    assert seen["remote_name"] is None
