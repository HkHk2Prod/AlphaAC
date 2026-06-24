#!/usr/bin/env sh
set -eu
uv run --frozen aczero benchmark --config configs/experiments/benchmark_rank2.yaml
