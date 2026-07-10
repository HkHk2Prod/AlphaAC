"""Write a dataset/annotation Markdown summary and push it to the Hugging Face bucket.

``aczero dataset grow``/``annotate`` and the hand-run Kaggle notebooks share this
step: after producing a data file they write its summary and upload the data file
and summary together. Uploads are lenient -- a missing ``HF_TOKEN``, an absent
``huggingface_hub``, or a per-file hub error is recorded rather than raised, so an
offline run still finishes with its summary written to disk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ac_zero.datasets.hub import DEFAULT_BUCKET, upload_dataset
from ac_zero.datasets.summary import summary_remote_name

# ``write_dataset_summary`` / ``write_annotation_summary``: both ``(path, dir) -> Path``.
SummaryWriter = Callable[[str | Path, str | Path], Path]


@dataclass(frozen=True, slots=True)
class UploadOutcome:
    """One file's upload result: a ``uri`` on success, an ``error`` string on skip."""

    remote_name: str
    uri: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class PublishResult:
    """The summary written (if any) and the per-file upload outcomes."""

    summary_path: Path | None
    outcomes: list[UploadOutcome]

    @property
    def uploaded_uris(self) -> list[str]:
        return [o.uri for o in self.outcomes if o.uri is not None]


def _upload_each(pairs: list[tuple[str | Path, str]], *, bucket: str) -> list[UploadOutcome]:
    """Upload each ``(local, remote)`` pair on its own, recording rather than raising.

    A missing token, absent ``huggingface_hub``, or a per-file hub/IO error becomes
    an ``UploadOutcome`` with ``error`` set, so one bad file never aborts the batch.
    """
    outcomes: list[UploadOutcome] = []
    for local, remote in pairs:
        try:
            uri = upload_dataset(local, remote_name=remote, bucket=bucket)
            outcomes.append(UploadOutcome(remote, uri, None))
        except Exception as exc:  # report any hub/token/IO failure per file, never abort
            outcomes.append(UploadOutcome(remote, None, str(exc)))
    return outcomes


def publish_to_bucket(
    data_path: str | Path,
    *,
    summary_writer: SummaryWriter | None = None,
    summary_dir: str | Path | None = None,
    bucket: str = DEFAULT_BUCKET,
    upload: bool = True,
) -> PublishResult:
    """Write ``data_path``'s Markdown summary and upload the data file and summary.

    ``summary_writer`` is ``write_dataset_summary`` or ``write_annotation_summary``
    (both take ``(path, summary_dir)``); pass ``None`` to upload the data file
    alone. The data file uploads under its basename and the summary under
    ``datasets_summaries/<name>``. ``upload=False`` still writes the summary but
    skips the bucket, so a local-only run keeps its report. Uploads never raise;
    inspect the returned outcomes for per-file errors.
    """
    data_path = Path(data_path)
    summary_path: Path | None = None
    if summary_writer is not None:
        if summary_dir is None:
            raise ValueError("summary_dir is required when summary_writer is given")
        summary_path = summary_writer(data_path, summary_dir)
    if not upload:
        return PublishResult(summary_path, [])
    pairs: list[tuple[str | Path, str]] = [(data_path, data_path.name)]
    if summary_path is not None:
        pairs.append((summary_path, summary_remote_name(summary_path)))
    return PublishResult(summary_path, _upload_each(pairs, bucket=bucket))
