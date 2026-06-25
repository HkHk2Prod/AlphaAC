.PHONY: setup install verify test smoke greedy-rl lint typecheck train dataset-refine

# One-command environment setup on a fresh machine (requires `uv`).
setup:
	uv sync --frozen

# Alias for setup.
install: setup

# Fast end-to-end check that the install works: lint, types, tests, smoke path.
verify: lint typecheck test smoke

test:
	uv run --frozen pytest

smoke:
	uv run --frozen aczero smoke-test

greedy-rl:
	sh scripts/test_greedy_rl_agent.sh

lint:
	uv run --frozen ruff check .

typecheck:
	uv run --frozen mypy src

# Serious training run. Override CONFIG/SEED, e.g.
#   make train CONFIG=configs/experiments/alphazero_rank2_heavy.yaml SEED=0
CONFIG ?= configs/experiments/alphazero_rank2_heavy.yaml
SEED ?= 0
train:
	uv run --frozen aczero train --config $(CONFIG) --seed $(SEED)

# Deep dataset refinement. Override INPUT and any search budget, e.g.
#   make dataset-refine ARGS="--max-difficulty -1 --max-expansions 200000"
INPUT ?= data/generated/train_rank2.json
ARGS ?= --max-difficulty 12 --max-expansions 100000 --max-generated 1000000
dataset-refine:
	uv run --frozen aczero dataset improve --input $(INPUT) $(ARGS)
