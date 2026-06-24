#!/usr/bin/env sh
set -eu
python scripts/bootstrap.py --accelerator cpu
uv run --frozen aczero smoke-test
uv run --frozen aczero certificate verify runs/smoke/certificates/example.json
