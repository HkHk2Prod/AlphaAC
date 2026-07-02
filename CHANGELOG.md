# Changelog

## Unreleased

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
