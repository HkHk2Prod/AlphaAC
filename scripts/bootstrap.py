#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default=os.environ.get("ACZERO_ACCELERATOR", "auto"))
    args = parser.parse_args()
    if args.accelerator not in {"auto", "cpu", "cuda12", "cuda13", "tpu"}:
        parser.error("invalid accelerator")
    selected = args.accelerator
    source = "explicit"
    if selected == "auto":
        selected = "cpu"
        source = "fallback"
        if os.environ.get("TPU_NAME"):
            selected = "tpu"
            source = "environment"
        elif shutil.which("nvidia-smi") and (
            subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        ):
            selected = "cuda12"
            source = "nvidia-smi"
    print(
        json.dumps(
            {"requested": args.accelerator, "selected": selected, "source": source},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
