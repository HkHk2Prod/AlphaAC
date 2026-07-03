.PHONY: setup install verify test smoke greedy-rl lint typecheck train dataset-annotate \
	dataset-pull dataset-push

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

# Annotate a group dataset with distances under a move set. Override INPUT/ARGS, e.g.
#   make dataset-annotate ARGS="--moveset strict-ac --max-depth 64"
INPUT ?= data/generated/train.groups.json
ARGS ?= --moveset universal
dataset-annotate:
	uv run --frozen aczero dataset annotate --input $(INPUT) $(ARGS)

# Sync the training dataset with the Hugging Face bucket (needs ac-zero[hub] and
# an HF_TOKEN). `data/generated/` is gitignored, so pull before annotating/using it.
dataset-pull:
	uv run --frozen --extra hub aczero dataset download --output $(INPUT)

dataset-push:
	uv run --frozen --extra hub aczero dataset upload --input $(INPUT)
