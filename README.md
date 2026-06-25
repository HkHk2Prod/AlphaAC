# AC-Zero

AC-Zero is a research software repository for searching for independently
verifiable Andrews-Curtis transformation certificates. A successful verified
certificate can show that a particular balanced presentation is not an
Andrews-Curtis counterexample. A failed search is not evidence that a
presentation is a genuine counterexample, and this project does not prove or
disprove the Andrews-Curtis conjecture.

The initial action space is the ordinary strict Andrews-Curtis action catalog:

- `AC1(i,j)`: replace `r_i` by the freely reduced product `r_i r_j`, with `i != j`.
- `AC2(i)`: replace `r_i` by `r_i^-1`.
- `AC3(i,g)`: replace `r_i` by `g r_i g^-1` for one signed generator `g`.

For rank `n`, the catalog has `3n^2` primitive actions. Stable moves and macros
must expand to these primitive moves before they can appear in a mathematical
certificate.

## Install

On a fresh machine, install [uv](https://docs.astral.sh/uv/) (`curl -LsSf
https://astral.sh/uv/install.sh | sh`), then from the repo root:

```bash
make setup     # uv sync --frozen — creates .venv from the committed lockfile
make verify    # lint + typecheck + tests + smoke path, to confirm the install
```

`.python-version` pins the interpreter (3.12) and `uv.lock` pins every
dependency, so `make setup` reproduces the exact environment. No network access
is needed beyond fetching the locked wheels. The accelerator probe is optional:

```bash
python scripts/bootstrap.py --accelerator cpu   # reports the selected backend
```

The training implementation is CPU-first and pure-NumPy, so it runs anywhere
Python does; optional JAX extras are declared for future accelerator work. If you
prefer pip over uv, `pip install -e .` against Python 3.11–3.14 also works.

## Five-Minute Smoke Test

```bash
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest
uv run --frozen aczero smoke-test
uv run --frozen aczero certificate verify runs/smoke/certificates/example.json
```

The smoke command generates a tiny strict-AC-solvable instance, writes a
checkpoint metadata file, emits a primitive certificate, and verifies it through
the independent replay verifier. It also uses callback loggers to stream
terminal progress, write structured JSONL events, mirror text progress logs, and
render ASCII metric graphs during and after the run:

- `runs/smoke/logs/training_events.jsonl`
- `runs/smoke/logs/progress.log`
- `runs/smoke/artifacts/live_graphs.txt`
- `runs/smoke/artifacts/final_graphs.txt`

## Data And Certificates

Use the committed standard presentation examples:

```bash
uv run --frozen aczero solve --presentation data/examples/standard_rank_1.json
uv run --frozen aczero solve --presentation data/examples/standard_rank_2.json
uv run --frozen aczero solve --presentation data/examples/standard_rank_3.json
```

Generate a synthetic training dataset. Instances are deduplicated by content
hash, exclude the trivial presentation, and each carries a `difficulty` label
(an upper bound on its strict-AC solution length), so the set scales to large
counts without repetition:

```bash
uv run --frozen aczero dataset generate \
  --config configs/experiments/smoke.yaml \
  --output data/generated/smoke.json
```

Write the curated catalog of standard potential Andrews-Curtis counterexamples
(Akbulut-Kirby and Miller-Schupp series) to `data/candidates/`, kept separate
from training data to avoid leakage:

```bash
uv run --frozen aczero dataset candidates --output data/candidates/standard.json
```

Run the small benchmark report:

```bash
uv run --frozen aczero benchmark \
  --config configs/experiments/benchmark_rank2.yaml
```

Run the config-driven policy/value training pipeline:

```bash
uv run --frozen aczero train \
  --config configs/experiments/alphazero_rank2.yaml \
  --seed 0
```

The training command generates a solvable curriculum, collects MCTS visit-count
policy targets into replay, optimizes the architecture named by `model:` in the
config, writes JSON checkpoints and metrics, emits progress logs/ASCII graphs,
and verifies a fixture certificate. The registered architectures
(`linear_policy_value`, `residual_mlp`, `deepsets`, `gru`, `transformer`) are
genuine trainable NumPy models built on a small reverse-mode autodiff engine and
trained by exact gradient descent; see [docs/architectures.md](docs/architectures.md).
They are deterministic CPU baselines, not a claim of production-scale neural
performance.

Run the dedicated greedy RL agent test pipeline:

```bash
sh scripts/test_greedy_rl_agent.sh
```

Run a solve with any implemented agent. Greedy stops honestly at a local
minimum; greedy best-first explores a length-ordered frontier; breadth-first and
iterative-deepening return shortest (and, within their caps, provably optimal)
certificates; `puct` runs the model-guided PUCT search.

```bash
uv run --frozen aczero solve --agent greedy
uv run --frozen aczero solve --agent greedy-best-first
uv run --frozen aczero solve --agent breadth-first
uv run --frozen aczero solve --agent iterative-deepening
uv run --frozen aczero solve --agent puct
```

Validate a dataset against the schema (structure, label fields, and recomputed
content hashes):

```bash
uv run --frozen aczero dataset validate --input data/generated/train_rank2.json
```

Improve a dataset's labels by searching each entry for a better trivialization.
Updates are merge-only: a shorter known solution is never replaced by a longer
one and known triviality is never demoted, duplicates are merged by content
hash, and proven-optimal entries are skipped so repeated passes are cheap. The
file is rewritten atomically.

```bash
uv run --frozen aczero dataset improve \
  --input data/generated/train_rank2.json --search all --max-difficulty 8
```

Verify a certificate:

```bash
uv run --frozen aczero certificate verify runs/smoke/certificates/example.json
```

The certificate artifact, not a checkpoint, is the mathematical object of
interest. The verifier parses the initial presentation, replays only strict
primitive AC moves, freely reduces after each move, checks intermediate hashes,
and checks the configured goal predicate.

## Scaling Up For A Serious Run

Everything is CPU-only and deterministic, so a bigger run is just bigger numbers.

Harder dataset refinement — raise the per-entry search budget and the difficulty
gate (a negative gate searches every entry). Improvement is monotonic, so a
longer run can only add shorter/optimal labels:

```bash
make dataset-refine ARGS="--max-difficulty -1 --max-expansions 200000 --max-generated 2000000"
# or directly:
uv run --frozen aczero dataset improve \
  --input data/generated/train_rank2.json \
  --search all --max-difficulty 15 --max-expansions 200000 --max-generated 2000000
```

Harder RL — copy `configs/experiments/alphazero_rank2_heavy.yaml` and scale the
knobs: `model` (`residual_mlp`/`gru`/`transformer`), `rank`, `dataset.depth`,
`training.{iterations,episodes_per_iteration,optimizer_updates,batch_size,
replay_capacity,mcts_simulations,c_puct,learning_rate}`. Then:

```bash
make train CONFIG=configs/experiments/alphazero_rank2_heavy.yaml SEED=0
```

The committed `alphazero_rank2_heavy.yaml` is calibrated to **~2 hours on one
modern CPU core** (~73 s/iteration at 24 episodes × 256 simulations on depth-6
instances; ~50 MB peak RAM). Wall-clock scales ~linearly with `iterations ×
episodes_per_iteration × mcts_simulations`, so scale `iterations` to your machine
and budget. Each run writes a reproducibility manifest (lockfile, platform,
config, seed), JSON checkpoints (every `checkpoint_every` iterations),
`metrics.jsonl`, and progress logs/graphs under `training.run_directory`, so a
long run on another machine is fully auditable and resumable from its checkpoint.

## Repository Layout

- `src/ac_zero/algebra`: immutable free-group words and balanced presentations.
- `src/ac_zero/moves`: primitive AC moves and deterministic action catalogs.
- `src/ac_zero/environment`: finite-horizon single-agent search environment.
- `src/ac_zero/certificates`: certificate JSON and independent verification.
- `src/ac_zero/datasets`: guaranteed-solvable strict-AC synthetic curriculum.
- `src/ac_zero/agents` and `src/ac_zero/search`: baseline agents and small MCTS.
- `src/ac_zero/models`, `encoding`, `training`: trainable policy/value
  architectures, a reverse-mode autodiff engine, replay, losses, smoke helpers,
  and the CPU policy/value training pipeline.
- `configs`, `data`, `docs`, `scripts`, `tests`: reproducibility and validation.

## Adding New Components

Add a presentation by writing canonical signed-integer relators with provenance
metadata. Do not mark an arbitrary input as trivial unless the provenance
supports that claim.

Add a model by subclassing `TrainablePolicyValueModel` (implement `_build_trunk`
and `_forward_trunk`) and registering its name in `models/registry.py`, or
implement the `PolicyValueModel` protocol directly for a non-trained baseline.
Add an agent by returning legal masked actions and a
structured `SolverResult`. Add a reward strategy only if its ablation label is
clear and the canonical telescoping reward remains available. Add a macro by
expanding it to strict primitive moves before certificate output.

## Limits

This repository provides a runnable research-grade foundation, a CPU smoke path,
and a deterministic CPU policy/value training baseline. The registered
architectures (`residual_mlp`, `deepsets`, `gru`, `transformer`, and the linear
baseline) are genuine trainable NumPy models trained end-to-end by exact gradient
descent. They are intentionally small CPU baselines; production-scale JAX/Flax
AlphaZero, PPO, and DQN systems on accelerators remain future work. Numerical
nondeterminism is expected on future GPU/TPU training runs; the manifest
machinery records lockfile, platform, configuration, and backend metadata to
make such differences auditable.
