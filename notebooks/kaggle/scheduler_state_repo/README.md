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
  queue.yaml                # desired tasks + mutable knobs (active/remaining_runs/priority)
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

> **Reliability note:** the HF bucket is a lightweight file-backed state store,
> not a transactional database. Buckets are last-writer-wins (no commit-SHA
> guard), so overlap is prevented by the scheduler **lease** file plus GitHub
> Actions `concurrency` — this is best-effort; inspect state from the Actions
> logs. (A dataset-repo backend additionally uses an optimistic parent-commit
> guard.)
