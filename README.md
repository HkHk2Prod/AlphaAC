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

### The ball: closest-first generation

`grow` orders the frontier by relator *length*, which is not the quantity training
cares about. The distances of what it produces have to be searched for afterwards
(`dataset annotate`), and even then they are only upper bounds — a group's true
shortest path to the origin may run through a group the run never expanded. Under
`strict-ac`, only 59% of the grown rank-2 dataset can reach the origin at all.

`dataset ball` inverts the construction: it walks outward from the trivial group by
the **inverses** of a move set's moves, breadth-first, so a group is discovered
exactly when the first inverse-move path reaches it — and that path, reversed, is a
shortest path of forward moves back to the origin.

```bash
uv run --frozen aczero dataset ball \
  --rank 2 --moveset strict-ac --max-relator-length 48 --target 1000000
# -> data/generated/ball_rank2_rel48.groups.json
```

Two properties follow, and they are the point:

* **Exact distances.** Every `distance_to_origin` is a proven optimum, so no
  annotation pass follows: the run writes `<name>.<moveset>.annotations.json` itself,
  with the co-optimal first moves a supervised policy is trained on.
* **Complete shells.** Groups are expanded in discovery order, so once the last group
  at distance `d` is expanded, *every* group at distance `d+1` is in the dataset. The
  deepest such `d` is reported as `complete_depth`.

Shells grow by roughly sevenfold a layer at rank 2 (1, 8, 51, 328, 2128, 14240,
97668, 683413, …), so a run is bounded by a group budget (`--target`, `--minutes`)
rather than by a depth, and resumes into the next shell on the next run.

`--max-relator-length` bounds the ball the same way the environment bounds an episode:
a move that would make a relator longer than the bound is one the environment masks, so
a group carrying an over-long relator is not in the graph at all. That is what keeps the
distances exact *for the model trained on them* — they are shortest paths through the
very graph the model moves in, not through long groups its encoder could never hold and
its environment would never let it enter. Only each relator is bounded; their sum is
not. The bound is part of the dataset's identity: it is recorded in the file, carried in
its name, and a training run whose `max_relator_tokens` disagrees with it is refused.
`0` grows the ball unbounded — the whole graph, with no model attached.

A bounded ball is therefore the default dataset for training runs
(`ball_rank{rank}_rel{max_relator_tokens}.groups.json`), and **a model of a different
capacity trains on a different ball**, not on a filtered view of the same one. The
length-first `grow` dataset remains for the universal move set and descent labels.

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

Run the small cross-solver regression report:

```bash
uv run --frozen aczero benchmark solvers
```

## Benchmark Evaluation

Separate from that regression, `aczero benchmark create` / `run` score a *trained
model* against the Akbulut-Kirby and Miller-Schupp series. Two bounds define the
catalog:

* `--max-relator-length` — no relator may be longer than this after free
  reduction. It bounds both families' index (AK(n) has a relator of length
  `2n+1`, MS(n, w) one of length `2n+3`). Match it to the training tasks'
  `max_relator_tokens` so the benchmark only asks models about presentations
  their encoder can represent.
* `--max-w-length` — a separate cap on the Miller-Schupp word `w`. It is needed
  because the relator bound alone does not bound the sweep usefully: freely
  reduced words with x-exponent sum zero grow roughly threefold per letter, so
  the ~47 letters a bound of 48 permits describe more words than could ever be
  enumerated.

```bash
# ~14.3k presentations: AK(1..23) plus MS(n<=22, |w|<=7), deduplicated by content.
# Writes it locally AND publishes it to benchmark_datasets/ on the bucket; pass
# --no-upload to keep it local.
uv run --frozen aczero benchmark create --max-relator-length 48 --max-w-length 7

# Score a checkpoint lineage's best model against it, publishing to the bucket
uv run --frozen aczero benchmark run \
  --checkpoint-name rank2-ppo-residual_mlp-strict_ac-navigation-1a2b3c \
  --minutes 240 --upload
```

A run makes two passes: a cheap length-ordered classical **scan** over every
entry, then model-guided PUCT (the **deep** pass) on what the scan missed, in
smallest-first order. The wall-clock cap is checked between presentations, so a
truncated run keeps everything it scored. Without a checkpoint the scan runs
alone, which is the baseline a trained model's numbers are read against.

Budgets live in the queue task's `benchmark:` block (or a `--config` YAML) —
`scan_expansions`, `scan_generated`, `deep_simulations`, `deep_moves`,
`max_moves`. They are starting guesses, meant to be tuned from the first runs.

Results land in the bucket under `benchmarks/`, summaries at the top so the
folder listing is the leaderboard; the catalogs themselves live under
`benchmark_datasets/`, since a catalog is a shared *input* every model is scored
against, not one run's output:

```
benchmarks/
    <checkpoint_name>.json                # rolling summary, one file per model
    runs/<checkpoint_name>/<run_id>.json  # per-entry detail for one run
benchmark_datasets/
    <catalog_name>.json                   # the entry set, e.g. ak-ms-rel48-w7
```

The summary's `best_solved` is a high-water mark and `ever_solved` is the union
of every presentation the lineage has ever solved — a solve is a permanent fact
about a presentation, so a later run with less budget does not un-solve it.

On Kaggle this is automatic. Every scheduler tick reads each training task's
`model_checkpoints/<name>/index.json` and queues any best model whose metric has
reached the threshold (`BENCHMARK_METRIC_THRESHOLD`, default 0.30) — the
self-play success-rate EMA, the same number best models are selected by. The
`benchmark-ak-ms` task is blocked whenever nothing is pending, and on launch the
scheduler pops the highest-metric checkpoint and hands it to the run. A given
`(checkpoint_name, run_id)` is only ever evaluated once.

That threshold is the entry price, paid once. After it, a benchmark run is
expensive enough that a model has to have *meaningfully* improved to earn
another: it must close `BENCHMARK_ERROR_REDUCTION` (default 0.25) of its
remaining error, so a first evaluation at 0.35 puts the next rung at 0.51, then
0.63, 0.73, 0.79 — geometric in error, which is uniform in `log(1 - accuracy)`,
so every evaluation buys the same amount of evidence. The rung per checkpoint
name is kept in `ladder` in `queue/benchmark_queue.json`. Three cases sidestep
it, so the ladder can never quietly end the evaluations: a metric that is not an
accuracy (the return EMA off navigation runs) has no ladder and keeps the
one-evaluation-per-run behaviour; a model-format bump resets the rung, exactly as
best-model promotion does; and a rung goes stale after
`BENCHMARK_STALENESS_DAYS` (default 14), so a metric that plateaus just under the
next rung still gets a periodic data point.

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

**Split** the dataset into train/validation/test, writing a third companion file,
`<name>.split.json`, that syncs to the bucket like the others:

```bash
uv run --frozen aczero dataset split --input data/generated/train.groups.json
```

A group's split is a deterministic function of its content hash, so re-running
after a `dataset grow` assigns the newly added groups and **cannot move a group a
model has already been evaluated on**. The default is 80/10/10
(`--val-fraction` / `--test-fraction` change it). Note that the split separates
*groups*, not regions: in a graph grown from a single root, a validation group is
inevitably a neighbour of training groups, so this measures generalization to
unseen groups rather than to an unseen part of the graph.

## Supervised Pretraining

The dataset already knows the answer to "which move gets closer to the trivial
group": the annotation pass records every group's distance to the origin, and the
group file records which group each move reaches. Joining the two scores every
move by the quantity the task is defined on —

```
delta[group, move] = distance(where the move lands) - distance(group)
```

— so `-1` is a move on a shortest path, `0` stalls, and `+1` or worse retreats.
`agent: supervised` trains a policy directly on that signal, with no self-play.
The policy target is `softmax(-delta / target_temperature)` over the moves whose
neighbour has a known distance (a move that leaves the annotated region gets zero
target mass but still competes in the softmax, so probability spent there is
probability taken from the moves that descend). The value head is regressed on
`2 * gamma**distance - 1` at the same time, so a pretrained critic is worth
something to an RL run rather than starting from noise.

The committed configs are one ladder over the same `rel48` rank-2 ball, named for the
model they train — `_pretrain` (a 0.8M transformer) and `_pretrain_mlp` (a residual MLP)
are the RL warm starts, one per architecture the RL pipeline runs; `_2m`, `_14m`, `_100m`
scale the transformer up towards using the model as the solver outright:

```bash
# Pretrain the two models the RL pipeline uses, then fine-tune each with RL
uv run --frozen aczero train --config configs/experiments/supervised_rel48_pretrain.yaml
uv run --frozen aczero train --config configs/experiments/supervised_rel48_pretrain_mlp.yaml

# Scale up: 2.2M runs on a CPU, 14.2M and 99.2M want a GPU
uv run --frozen aczero train --config configs/experiments/supervised_rel48_2m.yaml
uv run --frozen aczero train --config configs/experiments/supervised_rel48_14m.yaml
uv run --frozen aczero train --config configs/experiments/supervised_rel48_100m.yaml
```

The two `_pretrain` configs set `training.early_stopping_patience`, so a pretraining run
trains until its **validation descent accuracy stops improving** for that many epochs in a
row (rather than a fixed epoch count), then stops — the best checkpoint is always the one
kept, so stopping early loses nothing.

Both need the group file, its annotations, and a split, and a supervised run
provisions all three before it trains — you do not run `dataset split` by hand.
The groups and annotations are refreshed from the Hugging Face bucket only when the
local copy differs from it *by byte size*, so an up-to-date ball is not re-downloaded
but a stale one is. The split is a local artifact (it is never kept in the bucket):
it is regenerated whenever it is missing or was built from a different dataset —
tracked by a tiny `<name>.split.meta.json` provenance sidecar — and otherwise left
in place. Every one of these decisions is printed under the run's opening `dataset`
lines, so the log shows exactly what was refreshed, skipped, or rebuilt.

The first supervised run on a dataset also builds two memory-mapped sidecars beside it
— the descent-label store and the instance store — by streaming the whole ball and
applying every move to every group. On a 30M-group ball that is hundreds of millions of
move applications, so it is fanned out across `training.workers` processes (0 = every
physical core) and reports progress under a `sidecar` phase as it goes, rather than
sitting silent for the better part of an hour. The sidecars are cached and fingerprinted,
so every later run on the same dataset memory-maps them and starts at once.

Rather than pay that cost inside the first `train`, precompute it ahead of the run with
`dataset labels`, which provisions the dataset and builds both sidecars for a config —

```bash
uv run --frozen aczero dataset labels --config configs/experiments/supervised_rel48_100m.yaml
uv run --frozen aczero train         --config configs/experiments/supervised_rel48_100m.yaml
```

The prebuild is a one-time step per dataset: once its sidecars exist, that `train` and
every later run, seed, or fine-tune on the same ball starts immediately. `train` still
builds them on demand if you skip the prebuild.

A run writes the same artifacts as an RL run — run directory, `metrics.jsonl`,
plots, and the Hugging Face checkpoint bundle — so fine-tuning is just pointing an
AlphaZero/PPO config at the checkpoint it produced:

```yaml
training:
  warm_start: runs/train/supervised_rel48_pretrain/model_checkpoint/best.json
```

### Pretrain locally, fine-tune on Kaggle

The same handoff works across machines through Hugging Face, which is how the Kaggle RL
tasks pick up a warm start. Pretrain locally and push the checkpoint to the bucket:

```bash
uv run --frozen aczero train \
  --config configs/experiments/supervised_rel48_pretrain.yaml --upload-checkpoints
```

Each `_pretrain` config pins a fixed `training.checkpoint_name`
(`pretrained-rank2-transformer-rel48`, `pretrained-rank2-residual_mlp-rel48`), so the
model always lands at the same bucket prefix. A Kaggle RL task then names that string in
`training.pretrained_checkpoint`:

```yaml
training:
  pretrained_checkpoint: pretrained-rank2-transformer-rel48
```

On the task's **first** run — when it has no checkpoint of its own on the bucket yet — RL
seeds from that pretrained model. On every **later** run it finds the task's own RL
checkpoint and resumes from that instead: the RL checkpoint wins once it exists, so the
pretrained model only ever seeds run one. The pretrained model and the RL task must share
`model`/`model_config` and `max_relator_tokens` for the weights to load, so the RL task's
`model_config` is set to match its `_pretrain` config's.

Each epoch is `training.optimizer_updates` minibatches of Adam, scored afterwards
on a fixed sample of the validation split. The metrics are stated in terms of the
distance, not the loss: **`val_descent_accuracy`** is how often the model's
top-ranked move actually reduces the distance to the origin, and
**`val_mean_delta`** is the average distance change that move causes. The best
checkpoint is chosen by descent accuracy. Every trainable group has at least one
descending move by construction, so a perfect model scores `1.0` and `-1.0`. The
test split is scored exactly once, after the final epoch, and never steers
training. `val_descent_accuracy` is also the early-stopping signal: with
`training.early_stopping_patience` set, a run stops once that many epochs pass without it
improving by at least `early_stopping_min_delta`. Unlike the RL loops, the supervised
stage trains on **every distance the dataset contains** — there is no difficulty ceiling
and no curriculum.

Only **expanded** groups carry a label. An unexpanded frontier group — one
discovered as a neighbour but never itself expanded — has no recorded transitions,
so there is no move to score and nothing to learn from it. On the current rank-2
dataset that is 343k labelled groups out of 3.57M (the rest are frontier), which
the run reports as `train`/`val`/`test` counts in its opening `dataset` event.
More `dataset grow` therefore widens the supervised training set only insofar as
it *expands* groups, not merely discovers them.

### Encoder capacity — the one length bound

`max_relator_tokens` is the longest relator anything in a run may carry, and it is a
single number wearing three hats: the encoder's grid width, the bound the environment
masks moves by, and the bound the dataset was generated under. Nothing bounds the
relators' *sum*.

The encoder lays every presentation on a fixed `(rank, max_relator_tokens)` grid, and a
relator too long to fit is an error rather than a silent truncation — a truncated
relator is a different, mathematically wrong presentation. Nothing ever needs
truncating, though, because the environment's legal-move mask enforces the same bound
(an episode cannot walk into a state the model would have to refuse) and the dataset was
grown under it (no group in it is one the grid cannot hold).

Because the three have to agree, a run **refuses to start on a dataset generated under a
different bound**. A ball grown to `rel48` proves shortest paths through the graph a
48-token model moves in; a 32-token model trained on it would chase descents through
groups it can neither represent nor legally reach. This is not a filter away — dropping
the offending groups leaves the *surviving* distances wrong, since they were proven over
paths that ran through the dropped ones. A fine-tuning RL config must likewise set the
same `max_relator_tokens` as the checkpoint it warm-starts from, since the capacity
fixes the network's input shape.

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

## Kaggle Run Scheduler

A GitHub Actions cron launches Kaggle notebook runs automatically, so generation,
annotation, and training keep making progress across the ~12 h Kaggle session cap
without anyone clicking *Run*. The scheduler stores its task queue and run state
**in the same private Hugging Face bucket as the training dataset**
(`HkHk2Prod/alphaac-data`) and ferries the HF model token into Kaggle through a
**private Kaggle dataset**.

```
GitHub Actions (every 2 h)  ->  scripts/kaggle_scheduler.py
   -> reads/writes the private HF bucket (queue.yaml, scheduler_state.json, runs/)
   -> updates the private Kaggle runtime-secrets dataset (hf_token.txt)
   -> `kaggle kernels push` the one unified notebook with a per-run runtime_config.json
        -> notebook reads config + HF token, runs the mode, heartbeats to the bucket
```

The state lives alongside the dataset in the bucket. To use a separate Hugging
Face **dataset repo** instead (which adds an optimistic parent-commit guard), set
`HF_STATE_REPO_TYPE: dataset` and an `HF_STATE_REPO_ID` in the workflow.

The moving parts live in [`src/ac_zero/scheduler/`](src/ac_zero/scheduler/) (pure,
tested logic), the entrypoints in [`scripts/`](scripts/), the workflow in
[`.github/workflows/kaggle_scheduler.yml`](.github/workflows/kaggle_scheduler.yml),
and the notebook + seed state files in
[`notebooks/kaggle/`](notebooks/kaggle/) (`scheduler_runner.ipynb`,
`kernel-metadata.json`, and `scheduler_state_repo/`).

### Required GitHub secrets

| Secret | Purpose |
| --- | --- |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API credentials (push kernels, update the secrets dataset). |
| `HF_TOKEN` | Default Hugging Face token, used for both purposes below if scoped for both. |
| `HF_MODEL_TOKEN` *(optional)* | Read access to private/gated HF **models**; falls back to `HF_TOKEN`. Copied into the Kaggle secrets dataset. |
| `HF_STATE_TOKEN` *(optional)* | Read+write on the HF **bucket** holding scheduler state; falls back to `HF_TOKEN`. |

**HF token permissions:** the model token needs *read* on the model repos you pull;
the state token needs *read+write* on the `alphaac-data` bucket namespace. One
`HF_TOKEN` with both permissions works for everything (including the notebook's
heartbeat writes, which reuse the token from the Kaggle dataset).

### One-time setup

1. **Seed the scheduler state into the bucket** from
   [`notebooks/kaggle/scheduler_state_repo/`](notebooks/kaggle/scheduler_state_repo/)
   (the bucket already exists — it holds the dataset):
   ```bash
   export HF_TOKEN=...   # write access to the bucket namespace
   cd notebooks/kaggle/scheduler_state_repo
   python -c "from ac_zero.datasets.hub import upload_files; \
     upload_files([('queue.yaml','queue.yaml'), \
                   ('scheduler_state.json','scheduler_state.json')], \
                  bucket='HkHk2Prod/alphaac-data')"
   ```
   The workflow already sets `HF_STATE_REPO_ID: HkHk2Prod/alphaac-data` with
   `HF_STATE_REPO_TYPE: bucket`.
2. **Publish the notebook once** so its slug exists, and set each task's
   `notebook_slug` in `queue.yaml` to it:
   ```bash
   cd notebooks/kaggle && kaggle kernels push -p .
   ```
   (The scheduler patches `kernel-metadata.json` per launch — GPU flag, privacy,
   the runtime-secrets input — so you only push manually to reserve the slug.)
3. Add the GitHub secrets above under *Settings → Secrets and variables → Actions*.

The first scheduled tick creates the private `<user>/runtime-secrets` Kaggle
dataset automatically.

### How `queue.yaml` works

`queue.yaml` is the human-editable source of truth for **what should run**. Each
task carries its definition (`mode`, `accelerator`, `notebook_slug`,
`notebook_dir`, `max_runtime_minutes`, `config`) plus the mutable scheduling knobs
`active`, `remaining_runs`, `priority`, and `stop_after_current_iteration`. Global
`limits` cap concurrency:

```yaml
limits:
  max_total_active: 5        # total live Kaggle runs
  max_cpu_active: 5
  max_gpu_active: 1
  max_launches_per_tick: 1   # how many to start per scheduler run
  stale_heartbeat_minutes: 180
```

The seed queue ships an active generation + annotation task and three
**deactivated** training tasks (`active: false`) you flip on when a dataset
exists — AlphaZero, PPO, and a greedy-RL baseline — all on the `strict-ac`
(AC-primitive) move set. The two learners use the `navigation` reward, so they
seed self-play from the grown dataset and its `strict-ac` annotations (pulled
from the bucket by the notebook at startup); the greedy task is a non-learning
solve. A training task's `config` **is** the training config: the notebook writes
it to a YAML and runs `aczero train --config …`, so no repo `configs/` files are
needed on Kaggle.

Task selection each tick: skip inactive / exhausted / already-running tasks, order
by highest `priority`, break ties by oldest last-launch then id, and launch up to
the free-slot and per-tick budget. The machine-owned
`scheduler_state.json` (written only by the controller) tracks each task's active
run id, timestamps, latest status, and last error.

### How `remaining_runs` works (and why failed runs count)

`remaining_runs` counts **launches, not successes**. It is decremented the moment
a Kaggle run is *launched*, and never restored if that run later fails — a failed
scheduled run is still a completed scheduled run. When it hits `0` the task is set
`active: false`. Use `remaining_runs: null` for **infinite** (run forever). A
launch that never happened (e.g. `kaggle kernels push` errored) does **not**
decrement.

### Manual control

Edit `queue.yaml` in the HF state repo (or `scheduler_state.json` for the global
flags), or use the workflow's manual trigger:

- **Activate / deactivate a task** — set `active: true|false`.
- **Set run budget** — set `remaining_runs` to an integer or `null` (infinite).
- **Reprioritize** — bump `priority`, or pass `task_id` on the manual trigger.
- **Force a launch** — manual trigger with `task_id` + `force: true` (launches even
  if that task already has an active run; slot limits still apply).
- **Pause everything** — set `scheduler_paused: true` in `scheduler_state.json`.
- **Drain (stop new launches, keep active runs)** — set `stop_launching: true`.
- **Stop a running task cleanly** — set the task's `stop_after_current_iteration:
  true` (or `active: false`); the notebook polls the queue, checkpoints, uploads,
  marks the run stopped, and exits.

Trigger a run manually from **Actions → Kaggle Scheduler → Run workflow**
(`task_id`, `force`, `dry_run`, `max_launches`), or locally:

```bash
HF_STATE_REPO_ID=HkHk2Prod/alphaac-data HF_STATE_REPO_TYPE=bucket HF_TOKEN=... \
  KAGGLE_USERNAME=... python scripts/kaggle_scheduler.py --dry-run
```

`--dry-run` (or the `dry_run` input) logs every decision and writes nothing.

### Rotating the HF token

Update the `HF_TOKEN` / `HF_MODEL_TOKEN` GitHub secret, then run the workflow (or
`scripts/update_kaggle_runtime_secrets.py`). It pushes a new version of the private
Kaggle secrets dataset (deleting old versions where the Kaggle CLI supports it) so
the next Kaggle run picks up the new token. Revoke the old token afterwards.

### Concurrency, safety, and caveats

Two lines of defence stop overlapping ticks: the GitHub Actions `concurrency`
group (only one workflow run at a time) and a lease file
(`locks/scheduler_lease.json`) re-checked before launching. The bucket backend is
last-writer-wins, so it leans on those two; the optional dataset-repo backend adds
an optimistic parent-commit guard (re-read on conflict). Secrets never touch the
notebook source, `runtime_config.json`, logs, checkpoints, or uploaded outputs.

> ⚠️ **Security:** the HF model token is stored as a plain file inside a *private*
> Kaggle dataset. This is a workaround, not a real secret manager — keep the
> dataset private, scope the token minimally, and rotate it if leaked.
>
> ⚠️ **Reliability:** the Hugging Face bucket is used as a lightweight file-backed
> state store, not a transactional database. The lease + `concurrency` (plus the
> dataset-repo backend's parent-SHA guard) prevent the common races, but this is
> best-effort — inspect state from the Actions logs.

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
- `src/ac_zero/benchmarks`: the AK/MS catalog enumerator, the two-pass evaluation
  run that scores a checkpoint against it, and the summary/detail documents it
  publishes under `benchmarks/`.
- `src/ac_zero/scheduler`: the GitHub Actions-driven Kaggle run scheduler
  (queue/state models, HF-backed state store with a lease, task selection, the
  Kaggle CLI wrapper, the controller tick, the benchmark evaluation queue, and
  the notebook-side run reporter).
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
