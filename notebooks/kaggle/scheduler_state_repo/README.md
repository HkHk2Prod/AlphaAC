# Scheduler state (in the shared HF bucket)

The AC-Zero Kaggle scheduler keeps its persistent state **in the same private HF
bucket as the training dataset** (`HkHk2Prod/alphaac-data`) — one place for
everything. These files are seeds; upload them into the bucket to bootstrap the
scheduler. (To use a separate dataset repo instead, set `HF_STATE_REPO_TYPE=dataset`
and `HF_STATE_REPO_ID=<user>/<repo>` in the workflow.)

## Layout (inside the bucket)

All scheduler state lives under a single `queue/` folder, kept apart from the
`datasets/` and `model_checkpoints/` trees that share the bucket:

```text
queue/
  queue.yaml                # desired tasks + mutable knobs (active/remaining_runs/priority/start_fresh)
  scheduler_state.json      # machine-owned runtime state (controller is the only writer)
  runtime_configs/latest/   # last runtime_config.json handed to a launch (audit; secret-free)
  runs/
    <run_id>.json           # per-run status + heartbeat written by the notebook
    latest.json             # most recent run record
  locks/
    scheduler_lease.json    # scheduler lease (second line of defence vs overlap)
```

## Bootstrap

```bash
export HF_TOKEN=...   # write access to the bucket namespace
cd notebooks/kaggle/scheduler_state_repo
python -c "from ac_zero.datasets.hub import upload_files; \
  upload_files([('queue.yaml','queue/queue.yaml'), \
                ('scheduler_state.json','queue/scheduler_state.json')], \
               bucket='HkHk2Prod/alphaac-data')"
```

`HF_STATE_REPO_ID` in `.github/workflows/kaggle_scheduler.yml` already points at
`HkHk2Prod/alphaac-data` with `HF_STATE_REPO_TYPE: bucket`.

## Restarting a model from scratch

A training task normally resumes: it warm-starts from the best model already under
`model_checkpoints/<name>/`. Two one-shot flags in `queue.yaml` break that:

* `start_fresh: true` on a task — its next launch moves that name's whole bucket tree
  to `model_checkpoints/_archive/<name>/<timestamp>/` and re-seeds from the task's
  `pretrained_checkpoint` (or from zero when it names none). Nothing is deleted; the
  old runs stay readable under the archive path.
* `start_fresh_all: true` at the top of the file — the next tick sets `start_fresh` on
  *every* task and clears itself. This is applied before any other scheduling work, so
  a task launched in that same tick already carries the restart. It also empties
  `queue/benchmark_queue.json` (pending evaluations, dispatch history and ladder rungs):
  every entry there points at a lineage being archived, so the evaluation queue starts
  over with the models and that tick runs no benchmark-gate scan.

Both are consumed by the launch and written back as `false`, so setting one restarts a
task exactly once rather than on every tick. A launch that fails to push keeps the flag:
nothing was archived, so the restart is still owed.

## Comparing models

Each run's checkpoint upload also drops its all-runs figures into a bucket-wide
comparison tree, one folder per plot type and one file per model:

```text
plots/
  selfplay_progress/<checkpoint_name>.png
  loss_curves/<checkpoint_name>.png
  shaping_alpha/<checkpoint_name>.png
  validation/<checkpoint_name>.png
```

so comparing every model in the queue on one metric is opening one folder, rather than
opening the same figure inside each `model_checkpoints/<name>/plots/all_runs/`.

> **Reliability note:** the HF bucket is a lightweight file-backed state store,
> not a transactional database. Buckets are last-writer-wins (no commit-SHA
> guard), so overlap is prevented by the scheduler **lease** file plus GitHub
> Actions `concurrency` — this is best-effort; inspect state from the Actions
> logs. (A dataset-repo backend additionally uses an optimistic parent-commit
> guard.)
