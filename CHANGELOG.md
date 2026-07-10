# Changelog

## Unreleased

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
