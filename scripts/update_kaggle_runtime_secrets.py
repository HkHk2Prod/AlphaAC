#!/usr/bin/env python3
"""Push the HF model token into a private Kaggle dataset for the notebook.

Kaggle notebooks cannot read GitHub Actions secrets, so the scheduler ferries
the Hugging Face **model** token through a private Kaggle dataset
(``<user>/runtime-secrets``) as ``hf_token.txt``. The notebook reads it from
``/kaggle/input/runtime-secrets/hf_token.txt``.

SECURITY: this is a workaround, not a secret manager. The token lives as a plain
file inside a *private* Kaggle dataset. Keep the dataset private (never
``--public``), scope the token minimally, and rotate it if leaked. This script
never prints the token.

Environment:
  HF_MODEL_TOKEN / HF_TOKEN   token to publish (model token preferred)
  KAGGLE_USERNAME             owner of the dataset
  KAGGLE_SECRETS_DATASET      override the ``<user>/runtime-secrets`` slug
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def _resolve_slug() -> str:
    slug = os.environ.get("KAGGLE_SECRETS_DATASET", "").strip()
    if slug:
        return slug
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not user:
        raise SystemExit("ERROR: set KAGGLE_USERNAME or KAGGLE_SECRETS_DATASET.")
    return f"{user}/runtime-secrets"


def _resolve_token() -> str:
    token = (os.environ.get("HF_MODEL_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit("ERROR: set HF_MODEL_TOKEN or HF_TOKEN with the HF model token.")
    if not token.startswith("hf_"):
        raise SystemExit("ERROR: token does not look like a Hugging Face token (no 'hf_' prefix).")
    return token


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)


def _publish(work: Path, slug: str, title: str) -> None:
    """Create the dataset on first run, otherwise push a new version."""
    (work / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": slug, "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )
    base = ["kaggle", "datasets", "version", "-p", str(work),
            "-m", "rotate hf token", "--dir-mode", "zip"]
    # A new version keeps the same slug; --delete-old-versions prunes stale token
    # copies (needs a recent Kaggle CLI; ignored/failed calls fall back to create).
    version = _run([*base, "--delete-old-versions"])
    if version.returncode == 0:
        print(f"Updated private Kaggle dataset {slug} (new version).")
        return

    combined = (version.stdout + version.stderr).lower()
    if any(flag in combined for flag in ("delete-old-versions", "unrecognized", "no such option")):
        version = _run(base)
        if version.returncode == 0:
            print(
                f"Updated private Kaggle dataset {slug} (new version). NOTE: installed Kaggle "
                "CLI cannot delete old versions; old token versions accumulate."
            )
            return
        combined = (version.stdout + version.stderr).lower()

    if "not found" in combined or "404" in combined or "does not exist" in combined:
        create = _run(["kaggle", "datasets", "create", "-p", str(work), "--dir-mode", "zip"])
        if create.returncode == 0:
            print(f"Created private Kaggle dataset {slug}.")
            return
        raise SystemExit(f"ERROR: failed to create Kaggle dataset {slug}: {create.stderr.strip()}")

    raise SystemExit(f"ERROR: failed to update Kaggle dataset {slug}: {version.stderr.strip()}")


def main() -> int:
    slug = _resolve_slug()
    token = _resolve_token()
    title = slug.split("/", 1)[-1]

    work = Path(tempfile.mkdtemp(prefix="runtime-secrets-"))
    token_file = work / "hf_token.txt"
    try:
        token_file.write_text(token + "\n", encoding="utf-8")
        os.chmod(token_file, 0o600)
        _publish(work, slug, title)
    finally:
        # Wipe the plaintext token from the runner's disk.
        if token_file.exists():
            token_file.unlink()
        for extra in work.glob("*"):
            extra.unlink()
        work.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
