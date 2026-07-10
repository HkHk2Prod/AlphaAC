# Kaggle notebooks

Three self-contained notebooks for running AC-Zero on [Kaggle](https://www.kaggle.com/code).
Each installs the project **from GitHub**, respects the ~12 h Kaggle session
limit (stopping with a safety margin so results are always flushed), and writes
its output to `/kaggle/working` (saved when you *Save Version*).

| Notebook | What it does | Output in `/kaggle/working` |
| --- | --- | --- |
| [`01_generate_dataset.ipynb`](01_generate_dataset.ipynb) | Grows a persistent, guaranteed-solvable AC dataset (`aczero dataset grow`) until the time budget is nearly spent, then writes a Markdown summary and publishes it to the bucket's `datasets_summaries/` folder. | `train_rank<N>.groups.json` (the dataset), `train_rank<N>.groups.summary.md` |
| [`02_annotate_dataset.ipynb`](02_annotate_dataset.ipynb) | Pulls the grown dataset **read-only** from the bucket and runs `aczero dataset annotate` for each move set in `ANNOTATE_MOVESETS` (default `universal` + `strict-ac`), warm-starting from any existing annotation file, then writes a per-file summary and publishes it to `datasets_summaries/`. Designed to run **concurrently** with generation, in a separate session — it never modifies or re-uploads the group dataset. | `train_rank<N>.<moveset>.annotations.json` and `…annotations.summary.md` per configured move set |
| [`03_train.ipynb`](03_train.ipynb) | Pulls the grown dataset and its `strict-ac` annotations from the Hugging Face bucket and runs the policy/value training pipeline (`aczero train`, AlphaZero by default), **seeding self-play from the dataset's instances**, under a wall-clock deadline. | `training_report.md`, `training_report.json`, `loss_*.png`, `selfplay.png`, and the full `run_<backend>_rank2/` (checkpoints, logs, metrics) |

## Running on Kaggle

1. **New Notebook** → *File → Import Notebook* and upload the `.ipynb`, or paste the cells.
2. In the sidebar **Settings**:
   - **Internet: On** — required for the GitHub `pip install`.
   - **Accelerator: None (CPU)** — enough for all three notebooks; the models are
     small and self-play/search fan out across CPU cores. A GPU is not needed.
3. Adjust the **Configuration** cell (time budget, rank, backend, …), then *Run All*.
4. When it finishes, *Save Version* to persist `/kaggle/working` as the notebook
   output (and, for generation, *Save Version → create a Dataset* to reuse it).

Generation and annotation are independent processes reading/writing disjoint
files, so `01_generate_dataset` and `02_annotate_dataset` can run **at the same
time** in two separate Kaggle sessions — annotation just needs generation to
have published at least one version of the dataset first.

## Time budget

All three notebooks take a `TIME_BUDGET_HOURS` (default `11.5`) and a
`SAFETY_MARGIN_MIN`. They stop cleanly once the remaining time drops below the
margin:

- **Generation** grows in chunks and checkpoints to disk between rounds, so the
  dataset on disk is always consistent — an interrupted run loses at most the
  groups added since the last checkpoint.
- **Annotation** checkpoints each move-set pass to disk every `CHECKPOINT_EVERY`
  freshly settled groups, and moves to the next move set (or stops) cleanly at
  the deadline.
- **Training** stops at the next iteration boundary via a deadline callback,
  keeping the last checkpoint and the streamed `training_events.jsonl`, from
  which the summary and plots are reconstructed.

## Install source / branch

All three notebooks default to `REPO_BRANCH = "main"`. `main` provides
`dataset grow`, the `dataset annotate` distance pass (distance-to-origin and the
descent difficulty label under a chosen move set), and `train`.

## Resuming generation and annotation beyond one session

Grow is resumable via the **Hugging Face bucket** (`HkHk2Prod/alphaac-data`). The
generation notebook pulls the current dataset at start and pushes the grown one
back when the session ends, so each 12 h session continues the last — no manual
Kaggle-dataset attaching needed. Add a Kaggle secret named **`HF_TOKEN`** (with
write access to the bucket namespace) under *Add-ons → Secrets*, and keep
`HF_DOWNLOAD_ON_START` / `HF_UPLOAD_ON_FINISH` set to `True` in the config cell.

Annotation resumes the same way, but per move set: it pulls the dataset
**read-only** (it never uploads it back — only generation does) plus any
existing `<name>.<moveset>.annotations.json` for a warm start, since `annotate()`
only recomputes groups whose shorter-distance is still unresolved. Pick the move
sets to run with `ANNOTATE_MOVESETS` (default `["universal", "strict-ac"]`).

Training does not resume across sessions — one run is a single session up to the
time budget.

> Training **seeds its self-play from the grown dataset**: it pulls
> `train_rank<N>.json` and its `strict-ac` annotations from the same Hugging
> Face bucket at start (add the `HF_TOKEN` secret for a private bucket) and
> starts each episode from one of the dataset's guaranteed-solvable
> presentations. Set `HF_DOWNLOAD_ON_START = False` to fall back to random
> scrambles.
