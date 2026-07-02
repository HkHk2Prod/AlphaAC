# Kaggle notebooks

Two self-contained notebooks for running AC-Zero on [Kaggle](https://www.kaggle.com/code).
Each installs the project **from GitHub**, respects the ~12 h Kaggle session
limit (stopping with a safety margin so results are always flushed), and writes
its output to `/kaggle/working` (saved when you *Save Version*).

| Notebook | What it does | Output in `/kaggle/working` |
| --- | --- | --- |
| [`01_generate_dataset.ipynb`](01_generate_dataset.ipynb) | Grows a persistent, guaranteed-solvable AC dataset (`aczero dataset grow`) until the time budget is nearly spent. | `train_rank<N>.json` (the dataset), `dataset_summary.md`, `dataset_stats.json`, `hist_*.png` |
| [`02_train.ipynb`](02_train.ipynb) | Runs the policy/value training pipeline (`aczero train`, AlphaZero by default) under a wall-clock deadline. | `training_report.md`, `training_report.json`, `loss_*.png`, `selfplay.png`, and the full `run_<backend>_rank2/` (checkpoints, logs, metrics) |

## Running on Kaggle

1. **New Notebook** → *File → Import Notebook* and upload the `.ipynb`, or paste the cells.
2. In the sidebar **Settings**:
   - **Internet: On** — required for the GitHub `pip install`.
   - **Accelerator: None (CPU)** — enough for both notebooks; the models are small
     and self-play/search fan out across CPU cores. A GPU is not needed.
3. Adjust the **Configuration** cell (time budget, rank, backend, …), then *Run All*.
4. When it finishes, *Save Version* to persist `/kaggle/working` as the notebook
   output (and, for generation, *Save Version → create a Dataset* to reuse it).

## Time budget

Both notebooks take a `TIME_BUDGET_HOURS` (default `11.5`) and a
`SAFETY_MARGIN_MIN`. They stop cleanly once the remaining time drops below the
margin:

- **Generation** grows in chunks and checkpoints to disk between rounds, so the
  dataset on disk is always consistent — an interrupted run loses at most the
  groups added since the last checkpoint.
- **Training** stops at the next iteration boundary via a deadline callback,
  keeping the last checkpoint and the streamed `training_events.jsonl`, from
  which the summary and plots are reconstructed.

## Install source / branch

Both notebooks default to `REPO_BRANCH = "main"`. `main` provides
`dataset grow` and `train`. The `dataset descent` difficulty pass (and the grow
Markdown summary) live on a feature branch — set `REPO_BRANCH` to that branch and
`RUN_DESCENT = True` in the generation notebook once it is pushed.

## Resuming generation beyond one session

Grow is resumable via the **Hugging Face bucket** (`HkHk2Prod/alphaac-data`). The
generation notebook pulls the current dataset at start and pushes the grown one
back when the session ends, so each 12 h session continues the last — no manual
Kaggle-dataset attaching needed. Add a Kaggle secret named **`HF_TOKEN`** (with
write access to the bucket namespace) under *Add-ons → Secrets*, and keep
`HF_DOWNLOAD_ON_START` / `HF_UPLOAD_ON_FINISH` set to `True` in the config cell.

Training does not resume across sessions — one run is a single session up to the
time budget.

> Generation and training are **independent**: training self-plays its own
> instances and does not read the generated dataset.
