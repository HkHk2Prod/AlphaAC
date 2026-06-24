# Changelog

## Unreleased

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
