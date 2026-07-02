#!/usr/bin/env sh
set -eu

uv run --frozen pytest tests/unit/test_greedy.py tests/integration/test_cli_greedy_pipeline.py
uv run --frozen aczero dataset grow \
  --output data/generated/greedy_rl.json \
  --rank 2 --target 20 --workers 1
uv run --frozen aczero solve \
  --presentation data/generated/greedy_rl.json \
  --agent greedy
uv run --frozen aczero certificate verify runs/solve/certificates/solution.json
uv run --frozen aczero benchmark --config configs/experiments/greedy_rl.yaml
