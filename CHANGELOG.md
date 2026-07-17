# Changelog

## Unreleased

- **Pretrained models pipeline: pretrain locally, fine-tune on Kaggle from Hugging Face.**
  Supervised pretraining now stops when it stops learning: `training.early_stopping_patience`
  (with `early_stopping_min_delta`) ends a run once its validation descent accuracy has not
  improved for that many epochs in a row, instead of running a fixed epoch count. The two
  `_pretrain` configs (a transformer and a new `_pretrain_mlp` residual MLP — one per
  architecture the RL pipeline runs) pin a fixed `checkpoint_name`, so
  `aczero train --config … --upload-checkpoints` always lands the model at the same HF
  bucket prefix. A Kaggle RL task then names it in `training.pretrained_checkpoint`: on the
  task's first run — before it has a checkpoint of its own — RL seeds from that pretrained
  model; on every later run it resumes its own RL checkpoint instead (the RL checkpoint wins
  once it exists, so the pretrained model only ever seeds run one). The RL tasks'
  `model_config` is aligned to the matching `_pretrain` config so the weights load.

- **The supervised run is no longer silent.** It emitted a per-epoch event all along, but
  nothing rendered it: at the default `summary` verbosity the terminal sink is muted and
  `ConsoleSummaryLogger` only recognised RL's `self_play` iteration line plus a fixed list
  of milestone phases, and `epoch` was in neither — so an SL run printed its start banner
  and then nothing at all until the final graph, however many hours later. `self_play` and
  `epoch` now share the bundled progress line, rendered with the metrics the run actually
  has (validation accuracy and loss, not return and replay size), and `sidecar`, `model`,
  `test`, and the supervised `dataset` event get their own milestone lines. The sidecar
  build is the one that mattered most: on a large ball it runs for minutes before the first
  epoch, and with nothing on the terminal it read as a hang.
- **The supervised graphs show what the run is scored on.** The graph metrics and plot
  specs were RL-shaped, so an SL run graphed only its three training losses — while
  `val_descent_accuracy`, the score it selects its best checkpoint on, appeared in no graph
  at all. The validation scores now get an ASCII series and a `validation.png`, and the
  validation loss shares `loss_curves.png` with the training loss it is meant to be read
  against. Both lists are shared between the two kinds of run: a series a run never emits
  is simply left out of its graphs, so RL writes no validation figure and SL no self-play
  one.
- **A supervised run reports its own failure.** `execute()` wrapped the run in a bare
  `finally: close()` with no `except`, so unlike the RL pipeline a crash mid-epoch closed
  the logs and reached the user as a bare traceback, with nothing in the event log saying
  the run had failed. It now emits an `error` event and re-raises.
- **The supervised configs are a ladder, named for the model.**
  `supervised_pretrain.yaml` and `supervised_large.yaml` become
  `supervised_rel48_pretrain.yaml` (0.8M) and `supervised_rel48_100m.yaml` (99.2M), joined
  by `supervised_rel48_2m.yaml` (2.2M, CPU-usable) and `supervised_rel48_14m.yaml` (14.2M).
  The old pair named a role and a vibe; the rung between "underfits" and "wants a GPU" had
  nothing in it, and neither name said which ball it was cut for.
- **`dataset ball` checkpoints on the clock, and a checkpoint is a push.** The interval
  is now `--checkpoint-hours` (default 4) instead of a group count: what a checkpoint
  buys is a bound on the *work* an interruption can destroy, which is measured in time,
  while its cost is a full rewrite of both documents — which grows with the ball, where a
  group count does not. Each checkpoint also uploads both documents to the bucket, since
  a checkpoint left on a disk that dies with the machine (a Kaggle container, a spot
  instance) buys nothing at all.
- **Kaggle: 10-hour sessions, and no job is killed mid-flush any more.** Every task drops
  to `max_runtime_minutes: 600`. A terminated job is killed outright — nothing in
  `ac_zero` handles SIGTERM — so whatever it had not pushed dies with the container, and
  every long job now runs on `budget_min(margin)`: what is left of the session *now*, less
  the minutes it needs to flush. Measuring from *now* is the fix — a job is launched only
  after its dataset is pulled from the bucket, and for a multi-gigabyte ball that pull was
  quietly eating the flush margin, so the watchdog reached the job before the job reached
  its own soft deadline. The ball reserves 20 minutes to rewrite both documents and push
  them (it had none, and was therefore always SIGTERM'd, making its last checkpoint
  silently the only output of a session); training keeps its 10.
- **Kaggle training pushes on the same 4-hour cadence, and stops checkpointing every
  iteration.** `--upload-every-hours` is now passed explicitly (4, matching the ball), and
  the training tasks move from `checkpoint_every: 1` to `25`. The checkpoint event is not
  a disk-write knob — it is what the HF uploader and the self-play showcase hang off, and
  every save rewrites the run's whole metrics history — so one per iteration was pure
  waste, while disabling it would have pushed nothing at all.

- **One length bound, and the dataset carries it.** `max_relator_tokens` is now the
  only length limit in the project: the encoder's grid width, the bound the environment
  masks moves by, *and* the bound the training dataset was generated under. The
  environment's `total_length_cap` — a second, independent limit on the relators' sum —
  is removed, along with the `safety_cap` truncation it produced; nothing bounds the sum
  any more.
  - `dataset grow` / `dataset ball` take `--max-relator-length` (0 = unbounded, was
    `--total-length-cap`, which bounded the *total*). A move that would overshoot the
    bound is one the environment masks, so a bounded ball is exactly the graph a model
    of that encoder capacity can move in — which is what keeps its proven distances
    exact *for that model*, rather than shortest paths routed through long groups the
    model can neither hold nor legally reach.
  - The bound is part of a dataset's identity: recorded in the file (`bounds`), carried
    in its name (`ball_rank2_rel48.groups.json`), and checked when a run opens it. A
    training run whose `max_relator_tokens` disagrees with its dataset is **refused**,
    rather than silently filtering the groups that do not fit — dropping them would
    leave the surviving distances wrong, since they were proven over paths running
    through the dropped ones. Different capacity, different dataset.
  - `max_relator_tokens: 0` (derive the capacity from the data) is gone: the data is now
    generated *for* the capacity, so the capacity must be stated. Its default is 48.
  - The Kaggle scheduler generates what it trains on: `ball-main` grows the ball at
    `max_relator_length: 48` and every training task trains at `max_relator_tokens: 48`,
    so both resolve to `ball_rank2_rel48` in the bucket (the runner derives the name from
    the bound with the same helper the CLI uses). `alphazero_rank2_heavy` is set to the
    same 48, so the local heavy run shares that dataset and checkpoint lineage.
    **The unbounded `ball_rank2` already in the bucket cannot be reused**: a bounded ball
    is a different graph, so `ball-main` starts the `rel48` ball from the origin.

- **`aczero dataset ball`: closest-first generation with proven distances.** Where
  `dataset grow` expands the *shortest* group under every universal move, this walks
  outward from the trivial group by the **inverses** of one move set's moves, in
  breadth-first order. A group is therefore discovered exactly when the first
  inverse-move path reaches it, and that path reversed is a shortest path of forward
  moves back to the origin — so every `distance_to_origin` it writes is a proven
  optimum rather than the upper bound a search over a partially expanded graph can
  give, and once the last group at distance `d` is expanded, *every* group at distance
  `d+1` is present (reported as `complete_depth`). It needs no `dataset annotate` pass:
  the run emits the annotation file itself, with the co-optimal first moves. The
  motivation is the state of the grown dataset: under `strict-ac` only 59% of its 3.5M
  groups can reach the origin at all, and its shells are incomplete from distance 4 up
  (2,068 groups at distance 4 against a true 2,128; 43,657 at distance 6 against
  97,668). Runs are bounded by a group budget (`--target`, `--minutes`) because shells
  grow ~7x a layer at rank 2, they checkpoint and resume (the file is the queue — the
  expanded prefix is a single number in it), and they apply no length cap, since
  capping would reroute the shortest paths that run through a long group. Training runs
  now default to `ball_rank{rank}.groups.json`.

- **The supervised labels no longer need a stored move graph.** `SupervisedStore` now
  derives each group's neighbours by applying the move set's moves rather than reading
  the group file's `transitions` map, which lets it label a ball (which stores no
  adjacency — that map is ~85% of a grown file's bytes) and, on a grown dataset, labels
  the frontier groups whose transitions were never recorded and which were silently
  dropped before. The labels are also built *for* an encoder capacity: a move whose
  child overflows `max_relator_tokens`, like a no-op, is unplayable in the environment
  and is left unlabelled rather than taught, and a group the encoder cannot hold is
  dropped from the trainable rows instead of truncated — which is what makes an
  uncapped ball safe to train on. The capacity is a source of the sidecar, so changing
  it rebuilds the labels.

- **A supervised training stage** (`agent: supervised`), a third backend alongside
  `alphazero` and `ppo`. It learns the move that reduces a group's distance to the
  trivial group, taking its labels from data the dataset already holds: scoring each
  move's neighbour against the annotation file's distance-to-origin gives, per group
  and per move, `delta = distance(neighbour) - distance(group)`. The policy target is `softmax(-delta / target_temperature)` over
  the moves whose neighbour has a known distance (unlabelled moves get zero target
  mass but stay in the softmax denominator); the value head is regressed on
  `2 * gamma**distance - 1` at the same time, so an RL run warm-starting from the
  checkpoint gets a usable critic as well as a policy. Evaluation is stated in the
  task's own terms — `val_descent_accuracy` (how often the top-ranked move actually
  reduces the distance) and `val_mean_delta` (the average distance change it causes)
  — and the best checkpoint is chosen by descent accuracy, not by loss. The held-out
  test split is scored once, after the final epoch. The stage trains on every
  distance in the dataset: no `dataset.max_difficulty`, no curriculum. New
  `configs/experiments/supervised_pretrain.yaml` (a small model to fine-tune with RL)
  and `supervised_large.yaml` (a ~99M-parameter transformer to use directly).

- **`aczero dataset split`** writes a third companion file, `<name>.split.json`,
  assigning every group to train/val/test (80/10/10 by default) and publishing it to
  the Hugging Face bucket alongside the groups and annotations. A group's split is a
  deterministic function of its content hash, so re-running after a `dataset grow`
  places the new groups without moving any group a model was already evaluated on,
  and the file regenerates byte-for-byte from scratch.

- **The models are batched and device-aware.** Every trunk
  (`linear_policy_value`, `residual_mlp`, `deepsets`, `gru`, `transformer`) now
  consumes an `EncodedBatch` of stacked states and returns `(batch, actions)` logits
  and `(batch,)` values; search evaluates one state as a batch of one, and training
  pushes whole minibatches through in a single pass instead of looping per example.
  `train_batch` and `ppo_update` are true batched tensor ops. Models take a `device`
  (`cpu` / `cuda` / `auto`), which is what makes a 100M-parameter model trainable at
  all. The transformer gained `num_heads` (defaulting to `1`, which is exactly the
  previous single-head layer, so existing checkpoints are unaffected). The removed
  `policy_value_loss` helper had no remaining caller.

- **The encoder no longer silently truncates a relator.** A relator longer than
  `max_relator_tokens` now raises instead of being clipped to fit — a clipped relator
  is a different, mathematically wrong presentation. The environment's legal-move mask
  enforces the same bound, so an episode can never step into a state the model would
  have to refuse, and the training env is now built with the *run's* encoder rather
  than a default one. `max_relator_tokens: 0` (supervised only) derives the capacity
  from the dataset's longest relator and records the resolved value in the checkpoint,
  so a fine-tuning run can reconstruct the network's input shape.

- The `grow` length cap and `annotate` search depth can now be **disabled with a
  `0` sentinel**: `total_length_cap=0` admits neighbours of any relator length, and
  `max_depth=0` runs the shorter-distance search unbounded. The Kaggle generation
  and annotation notebooks set both to `0`, so a run is bounded only by its time
  budget rather than discarding long presentations or leaving deep shortenings
  unproven. The `--total-length-cap` / `--max-depth` CLI flags accept the same
  sentinel.

- Training now has a console **verbosity** knob so a run no longer floods the
  terminal. `TrainingPipelineConfig.verbosity` (YAML `training.verbosity`, or
  `aczero train --verbosity`) takes `verbose` (the historical per-event lines plus
  an ASCII graph re-rendered on every event), `summary` (the new default: one
  compact line per logged iteration that bundles the batch's episodes into
  return/success/loss, plus run milestones and the final graph printed once), or
  `quiet` (only start/stop milestones and warnings). Every level still writes the
  full `training_events.jsonl` and the live/final graph files, so nothing is lost
  on disk. The new `ConsoleSummaryLogger` provides the summary output;
  `TerminalProgressLogger`/`AsciiGraphLogger` gained `console` flags so the file
  mirror is kept without printing. `default_training_callbacks` maps the level to
  the right sinks. The Kaggle training notebook (`03_train.ipynb`) exposes the same
  knob as `VERBOSITY` and builds its callbacks through `default_training_callbacks`
  instead of hand-muting the sinks to `devnull`.

- The Kaggle generation and annotation notebooks now publish a Markdown summary to
  the Hugging Face bucket. Summaries live in their own `datasets_summaries/` folder,
  named after the dataset (`train_rank2.groups.summary.md`,
  `train_rank2.groups.strict-ac.annotations.summary.md`), so they are browsable on
  HF without downloading the dataset. Generation reuses the shared
  `write_dataset_summary` (its inline copy and the unused histogram plots are gone);
  annotation gains a new `write_annotation_summary` that reports distance-to-origin
  and descent-distance distributions plus the proven-vs-unresolved split per move
  set. Both go through `summary_remote_name`, the single source of the folder name.

- Added a `"navigation"` reward mode: an adaptive distance-shaping reward for the
  start-to-goal search. Each transition scores a terminal destination bonus
  proportional to the start-to-goal distance `L0`, an `alpha`-weighted
  distance-reduction shaping term, a flat move fee, and a per-episode revisit fee,
  all from a config-driven `RewardConfig` (no hard-coded constants). A stateful
  `RewardComputer` (in `ac_zero.environment.navigation_reward`) owns the
  within-episode state (visited set, running minimum distance). Distance is the
  group's exact `distance_to_origin` annotation; a presentation off the annotated
  graph has no distance, so — like the `potential` reward — the environment holds
  the last known distance as an anchor and defers an off-graph excursion's shaping
  credit until the search re-enters a known node (the goal counts as distance 0),
  never inventing a length proxy. The mode therefore requires a distance-annotated
  dataset (rejected at config validation otherwise), so the start distance `L0` and
  every credited descent are exact. `alpha` is constant within an episode and retuned between
  episodes by an `AlphaUpdater` from success/progress EMAs. The training pipeline
  holds one updater per run, keeps `alpha` constant across each iteration's batch,
  advances it in deterministic collection order (so parallel workers stay
  reproducible), logs per-episode alpha and the evaluation metrics
  (`success_rate`, `progress_rate`, `average_*_reward`, distances, revisits), and
  — for this mode — selects the best checkpoint by success rate rather than shaped
  return. Replay/PPO transitions retain the separated reward components so a buffer
  entry can be re-scored later. Configure via a `reward:` YAML block.

- Fixed scheduled Kaggle training runs, which ended after `training.iterations`
  (40) instead of filling their runtime budget, and never touched Hugging Face.
  `aczero train` gains `--minutes`, a soft wall-clock budget (mirroring `dataset
  grow`) that stops the loop at the next iteration boundary and still writes the
  checkpoint, plots, certificate, and summary; `TrainingPipelineSummary.iterations`
  now reports the iterations actually run. `scheduler_runner.ipynb` passes it --
  reserving head-room so the final upload is not cut short by the watchdog -- plus
  `--download-checkpoint`, `--upload-checkpoints`, and `--checkpoint-bucket`, so a
  scheduled run warm-starts from the lineage's best model on the bucket and pushes
  its own bundle back. `queue.yaml` no longer caps training iterations.
- Fixed `datasets.hub.download_file` silently returning a path to a file that was
  never written when the object is absent from the bucket: the hub skips missing
  files with a warning by default, so it now passes `raise_on_missing_files=True`.
- Added Hugging Face model checkpoints with warm start. Each run keeps a bundle
  (`<run>/model_checkpoint/`: `best.json`, `latest.json`, `metrics.jsonl`,
  `meta.json`) current, tracking the best model by an EMA of self-play mean
  return. `training.checkpoint_name` (or an auto name derived from the
  model/moveset/reward/rank identity) selects the destination
  `model_checkpoints/<name>/` in the HF bucket; `training.warm_start` initializes
  a run's weights from a saved checkpoint. `PeriodicCheckpointUploader` pushes the
  best model, per-run and all-runs progress plots, and an `index.json` rollup
  every few hours (and once at the end); `download_best_checkpoint` pulls the best
  model back for the next run. Wired into `03_train.ipynb` via `CHECKPOINT_NAME`,
  `WARM_START`, `HF_CHECKPOINT_UPLOAD`, and `HF_UPLOAD_EVERY_HOURS`; the summary
  cell shows both this run's plots and the all-runs aggregate.
- Added the `potential` reward mode: potential-based shaping toward the trivial
  group, where the potential is a presentation's `distance_to_origin` annotation.
  Steps between annotated states score `Phi(prev) - Phi(next)` plus the
  `goal_reward` bonus on the goal. Steps into the unannotated region score zero;
  the environment holds the exit potential and, on re-entering the known region
  (the goal counts as a known `Phi = 0`), credits the whole `Phi(exit) - Phi(entry)`
  change at once. Because the undiscounted return of a potential is path-length
  invariant, the reward discount `training.gamma` (default `0.99`) discounts the
  return to mildly prefer shorter paths -- and deferring off-graph credit to the
  later re-entry step lets that discount account for the excursion. Requires
  `dataset.annotations`, and seeds self-play only from groups whose distance to the
  trivial group is known.
- Unified the reward discount into a single `training.gamma` (default `0.99`)
  applied by every training pipeline: the AlphaZero return-to-go targets (for all
  reward modes, not only `potential`) and the PPO GAE returns/advantages. Replaces
  the former `potential_gamma` (AlphaZero, `potential` mode only) and `ppo_gamma`
  (PPO only); `ppo_lambda` is unchanged.
- Removed the `descent` reward mode (unstable in training). The reward modes are
  `length_reduction`, `sparse_goal`, `length_reduction_and_goal` (default), and
  `potential`. The dataset's `distance_to_shorter` / `shorter_proven` annotations
  are still produced by `dataset annotate`; they are simply no longer consumed by
  any reward path.
- `01_generate_dataset.ipynb` and `02_annotate_dataset.ipynb` now fail fast if
  `HF_UPLOAD_ON_FINISH` is on but no `HF_TOKEN` is available (env var or Kaggle
  secret), instead of running for the whole time budget and only then failing
  to publish. Also quieted the missing-secret lookup itself: it no longer
  prints a scary-looking error when the notebook's kernel simply never had the
  `HF_TOKEN` secret attached (Kaggle secrets are per-notebook, not per-account).
- Split annotation out of the Kaggle generation notebook into its own
  `notebooks/kaggle/02_annotate_dataset.ipynb` (generation is now
  `01_generate_dataset.ipynb`, training renumbered to `03_train.ipynb`) — a 12 h
  Kaggle session left no time for both grow and annotate together. The new
  notebook only ever reads the group dataset from the Hugging Face bucket and
  writes/publishes its own `<name>.<moveset>.annotations.json` files, so it can
  run concurrently with generation without altering the dataset it reads.
  Existing annotation files are pulled first for a warm start (`annotate()`
  skips groups already settled). `ANNOTATE_MOVESETS` picks which move sets to
  run, defaulting to `["universal", "strict-ac"]`.
- The grown dataset now lives in a Hugging Face storage bucket
  (`HkHk2Prod/alphaac-data`) instead of git — it outgrew GitHub's 100 MB per-file
  limit. `aczero dataset upload` / `download` push and pull it (optional
  `ac-zero[hub]` dependency, `HF_TOKEN` auth), `make dataset-pull` / `dataset-push`
  wrap them, and `data/generated/` is now gitignored. The Kaggle generation
  notebook resumes from the bucket at start and publishes the grown dataset back
  at the end.
- `aczero dataset descent` annotates each entry with its length-descent distance
  — the fewest AC moves that strictly shorten the presentation — as the training
  example-difficulty label (`descent_distance` / `descent_proven`, exact within a
  length/depth budget). Backed by a breadth-first `search/descent.py`; entries are
  searched easiest-first, proven answers are skipped on re-runs, and the pass
  checkpoints and writes atomically.
- Portable setup for fresh machines: pinned interpreter (`.python-version`),
  `make setup` / `make verify` targets, and a documented uv/pip install path.
- `aczero dataset improve` exposes per-entry search budgets (`--max-expansions`,
  `--max-generated`) and an unbounded difficulty gate (`--max-difficulty -1`) for
  harder refinement runs; `make dataset-refine` wraps it.
- Training exposes `c_puct` in the experiment config; `make train` and a heavier
  `alphazero_rank2_heavy.yaml` config support scaled RL runs.

- Added a real single-player PUCT MCTS (`search/puct.py`) guided by the policy/
  value model. Training self-play now uses it for visit-count targets (replacing
  the uniform placeholder), and it is exposed as the `puct` solve/benchmark agent.
- Added iterative-deepening DFS (`search/iterative_deepening.py`, `solve --agent
  iterative-deepening`): shortest-path certificates with linear memory.
- `aczero dataset validate` now performs real schema validation
  (`datasets/validation.py`): structure, label-field types, and recomputed
  content hashes.
- `aczero benchmark` now runs every implemented solver (random, greedy, greedy
  best-first, breadth-first, iterative-deepening, puct) and `benchmark_rank2.yaml`
  lists exactly those agents.

- Added uninformed breadth-first search (`search/breadth_first.py`,
  `aczero solve --agent breadth-first`): shortest-path certificates, with a
  proven-optimality flag when the search completes within its caps.
- Added `aczero dataset improve` (`datasets/update.py`): runs the search agents
  over a dataset and merges better trivializations into the labels. The merge is
  monotonic (`merge_labels`) — a shorter known solution never loses to a longer
  one, triviality is never demoted, and proven-optimal entries are skipped.
  Duplicates are merged by content hash and the file is written atomically.

- Added per-entry trivialization labels to every dataset and example
  (`ac_trivial`, `minimal_known_operations`, `optimal`; see
  `datasets/labels.py`). Generated instances are known AC-trivial with a known,
  non-optimal solution; standard examples are trivial with zero optimal
  operations; open candidates carry the unknown label.
- `generate_dataset` / `write_dataset` accept `depths=[...]` to span an
  easy-to-hard difficulty range in one flat, deduplicated set.

- Scaled dataset generation: `generate_dataset` / `write_dataset` deduplicate by
  content hash, exclude the trivial presentation, and label every instance with a
  `difficulty` (scramble depth, an upper bound on solution length). Datasets are
  now `aczero-dataset-v2`.
- Added a curated catalog of standard potential Andrews-Curtis counterexamples
  (Akbulut-Kirby and Miller-Schupp series) in `datasets/candidates.py`, the
  `aczero dataset candidates` CLI command, and a committed
  `data/candidates/standard.json`, kept separate from training data.

- Added a small reverse-mode autodiff engine (`models/autograd.py`) with
  finite-difference gradient checks.
- Replaced the uniform policy/value stubs with genuine trainable architectures
  (`residual_mlp`, `deepsets`, `gru`, `transformer`) and a linear baseline on a
  shared `TrainablePolicyValueModel` base, trained end-to-end by gradient descent.
- Training pipeline now honors the `model:` config field instead of always using
  the inline linear model, and serializes/restores architecture weights in
  checkpoints.
- Documented the architectures in `docs/architectures.md`.

## 0.1.0

- Initial research-grade scaffold with verified algebra, strict AC moves, certificates, datasets, CLI smoke path, and tests.
