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


def _out(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stdout + proc.stderr).strip()


def _publish(work: Path, slug: str, title: str) -> None:
    """Push a new version if the dataset exists, otherwise create it.

    Rather than fragile message-matching, this just tries each path in order and
    stops at the first success: version-with-prune -> plain version -> create.
    On the very first run the version calls fail (no dataset yet) and create
    succeeds; on later runs the first version call wins.
    """
    (work / "dataset-metadata.json").write_text(
        json.dumps({"title": title, "id": slug, "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )
    base = ["kaggle", "datasets", "version", "-p", str(work),
            "-m", "rotate hf token", "--dir-mode", "zip"]

    # --delete-old-versions prunes stale token copies (needs a recent Kaggle CLI).
    pruned = _run([*base, "--delete-old-versions"])
    if pruned.returncode == 0:
        print(f"Updated private Kaggle dataset {slug} (new version; old versions pruned).")
        return

    plain = _run(base)
    if plain.returncode == 0:
        print(
            f"Updated private Kaggle dataset {slug} (new version). NOTE: installed Kaggle CLI "
            "could not prune old versions; old token versions accumulate."
        )
        return

    # Neither version call worked -- the dataset most likely does not exist yet.
    create = _run(["kaggle", "datasets", "create", "-p", str(work), "--dir-mode", "zip"])
    if create.returncode == 0:
        print(f"Created private Kaggle dataset {slug}.")
        return

    raise SystemExit(
        f"ERROR: could not update or create Kaggle dataset {slug}.\n"
        f"--- `datasets version --delete-old-versions` ---\n{_out(pruned)}\n"
        f"--- `datasets version` ---\n{_out(plain)}\n"
        f"--- `datasets create` ---\n{_out(create)}"
    )


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
