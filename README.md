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

Grow a training dataset outward from the trivial group. Each run expands known
groups by every **universal** move and records each newly reached group together
with its full local adjacency — a `move_id -> target hash` transition map, keyed
by the universal move that produces each neighbour — deduplicated by content
hash. The universal moves are *invertible* (every move has an inverse move in the
set), so generation is pure graph construction with no reverse-path bookkeeping:
distances are recovered later by the annotation pass. The database only ever
grows: the first run bootstraps from the trivial presentation, and every later
run resumes from the accumulated frontier and appends more groups.

The group file (`<name>.groups.json`) stores each group in minimal form: content
hash, rank, known AC-triviality, source, the relators (lists of signed integers),
total length, and the transition map. Every group reachable from the trivial root
by AC moves is AC-trivial.

```bash
uv run --frozen aczero dataset grow \
  --output data/generated/train.groups.json \
  --rank 2 --target 1000 --select smallest
```

Re-run the command (or point `--input` at the same file) to expand it further.
`--select smallest` gives one deterministic canonical frontier; use `--select
weighted-random --seed N` so independent machines explore divergent regions.

The grown dataset outgrows GitHub's 100 MB per-file limit, so it is **not** kept
in git (`data/generated/` is gitignored) — it lives in a Hugging Face
[storage bucket](https://huggingface.co/docs/hub/storage-buckets)
(`HkHk2Prod/alphaac-data`). Install the optional dependency and authenticate once
(`pip install ac-zero[hub]`; set `HF_TOKEN` or run `hf auth login`), then pull or
push the current dataset:

Both the group file and its annotation files (below) sync by basename, so the
same commands move either kind:

```bash
# Download the current training set into data/generated/train.groups.json
uv run --frozen --extra hub aczero dataset download --output data/generated/train.groups.json

# Push a locally grown dataset (or an annotation file) back to the bucket
uv run --frozen --extra hub aczero dataset upload --input data/generated/train.groups.json
```

`uv run` re-syncs the environment per invocation, so pass `--extra hub` on the
command itself (it isn't enough to `uv sync --extra hub` once).

`--bucket` and `--remote-name` override the defaults. The Kaggle generation
notebook does this automatically — it pulls the dataset at start (resume) and
uploads the grown one when the session ends (see
[notebooks/kaggle/](notebooks/kaggle/)).

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
and verifies a fixture certificate. When it finishes it renders PNG plots of the
training progress (`artifacts/loss_curves.png` and `artifacts/selfplay_progress.png`)
and reports their paths; on an interactive terminal with a display they are also
opened in the default image viewer. The registered architectures
(`linear_policy_value`, `residual_mlp`, `deepsets`, `gru`, `transformer`) are
genuine trainable NumPy models built on a small reverse-mode autodiff engine and
trained by exact gradient descent; see [docs/architectures.md](docs/architectures.md).
They are deterministic CPU baselines, not a claim of production-scale neural
performance.

The same pipeline also offers an on-policy PPO backend. Select it with
`agent: ppo` in a train config (see
[configs/experiments/ppo_rank2.yaml](configs/experiments/ppo_rank2.yaml)):

```bash
uv run --frozen aczero train --config configs/experiments/ppo_rank2.yaml --seed 0
```

Instead of MCTS self-play it samples rollouts from the current policy, estimates
advantages with GAE(`gamma`, `ppo_lambda`), and runs `ppo_epochs` of
minibatch clipped-surrogate updates (`ppo_clip`, `entropy_coef`, reusing
`value_loss_weight` for the value term) over that on-policy data — writing the
same checkpoints, metrics, plots, and verified certificate. Rollout collection
fans out across worker processes exactly like self-play, so the trained model is
independent of the worker count. A trained checkpoint can then drive a greedy
policy decode via `aczero solve --agent ppo --checkpoint <path>`.

Run the dedicated greedy RL agent test pipeline:

```bash
sh scripts/test_greedy_rl_agent.sh
```

Run a solve with any implemented agent. Greedy stops honestly at a local
minimum; greedy best-first explores a length-ordered frontier; breadth-first and
iterative-deepening return shortest (and, within their caps, provably optimal)
certificates; `puct` runs the model-guided PUCT search; `ppo` greedily decodes a
policy-value model (add `--checkpoint <path>` to use a PPO-trained one).

```bash
uv run --frozen aczero solve --agent greedy
uv run --frozen aczero solve --agent greedy-best-first
uv run --frozen aczero solve --agent breadth-first
uv run --frozen aczero solve --agent iterative-deepening
uv run --frozen aczero solve --agent puct
uv run --frozen aczero solve --agent ppo --checkpoint runs/train/ppo_rank2/checkpoints/latest.json
```

Validate a group or annotation dataset against its schema (structure, recomputed
content hashes for groups, distance/move-list fields for annotations):

```bash
uv run --frozen aczero dataset validate --input data/generated/train.groups.json
```

**Annotate** a group dataset with distances under a chosen move set, writing a
separate `<name>.<moveset>.annotations.json` file. Generation and annotation are
independent processes: generation builds the graph once, and each annotation pass
reads that stored adjacency. Available move sets are `universal` (the full
invertible set) and `strict-ac` (the classic `3n^2` catalog); `strict-ac` is a
subset of `universal`.

```bash
uv run --frozen aczero dataset annotate \
  --input data/generated/train.groups.json --moveset strict-ac
```

Each annotation entry carries, under that move set: the **distance to the origin**
(the trivial group) with the co-optimal first moves toward it, and the **distance
to a strictly shorter group** (the descent-distance difficulty label) with its
co-optimal first moves. Because the moves are invertible, distance-to-origin is a
single breadth-first sweep from the trivial root over the *inverse* move set —
which also makes annotating a non-invertible subset well-defined. Groups are
processed shortest-first, settled answers are skipped on re-runs, and the file is
rewritten atomically.

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

Bigger annotation runs — raise the descent search depth so more plateaus resolve
within budget. Annotation is exact where it terminates, so a longer run can only
settle more groups:

```bash
uv run --frozen aczero dataset annotate \
  --input data/generated/train.groups.json \
  --moveset strict-ac --max-depth 64 --workers 0
```

Generation, annotation, and training are all CPU-bound, so they fan independent
work across worker *processes* (threads cannot help under the GIL). This is on by
default: every CPU core is used unless you say otherwise. Pass `--workers N` to
`dataset grow` (database expansion) or `dataset annotate` (per-group descent
searches), or set `training.workers` in a train config (self-play episodes), where
`0` (the default) autodetects and uses every physical core (hyperthreads excluded,
since they add little for CPU-bound work), a negative count leaves that many free,
and `1` stays single-process. Results are reassembled in input order,
so the generated and refined datasets and the trained model are bit-for-bit
identical regardless of the worker count — parallelism trades cores for wall-clock
without touching reproducibility.

Harder RL — copy `configs/experiments/alphazero_rank2_heavy.yaml` and scale the
knobs: `model` (`residual_mlp`/`gru`/`transformer`), `rank`, `dataset.depth`,
`training.{iterations,episodes_per_iteration,optimizer_updates,batch_size,
replay_capacity,mcts_simulations,c_puct,learning_rate,workers}`. Then:

```bash
make train CONFIG=configs/experiments/alphazero_rank2_heavy.yaml SEED=0
```

The committed `alphazero_rank2_heavy.yaml` is **~2 hours on one modern CPU core**
(~73 s/iteration at 24 episodes × 256 simulations on depth-6 instances; ~50 MB
peak RAM per worker) and ships with `training.workers: 0`, which spreads the 24
self-play episodes across every core to cut that roughly by the core count.
Wall-clock otherwise scales ~linearly with `iterations × episodes_per_iteration ×
mcts_simulations`, so scale `iterations` to your machine and budget. Each run writes a reproducibility manifest (lockfile, platform,
config, seed), JSON checkpoints (every `checkpoint_every` iterations),
`metrics.jsonl`, and progress logs/graphs under `training.run_directory`, so a
long run on another machine is fully auditable and resumable from its checkpoint.

Warm starts across runs — a run also keeps a Hugging Face-shaped bundle under
`<run_directory>/model_checkpoint/` (`best.json`, `latest.json`, `metrics.jsonl`,
`meta.json`), tracking the best model by an EMA of self-play mean return. Set
`training.checkpoint_name` (or let it auto-derive from the model/moveset/reward/
rank identity, so the same setup resumes one lineage) and `training.warm_start`
(a local checkpoint whose weights initialize the run). The Kaggle training
notebook pushes the best model, this-run and all-runs progress plots, and an
`index.json` rollup to `model_checkpoints/<name>/` in the HF bucket every few
hours, and pulls the best model back for the next run — a warm start toward
longer training split across sessions.

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
descent, with both AlphaZero (PUCT self-play) and PPO training backends. They are
intentionally small CPU baselines; production-scale JAX/Flax training on
accelerators, and a DQN backend, remain future work. Numerical
nondeterminism is expected on future GPU/TPU training runs; the manifest
machinery records lockfile, platform, configuration, and backend metadata to
make such differences auditable.
