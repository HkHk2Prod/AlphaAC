"""GitHub Actions-driven scheduler for automated Kaggle notebook runs.

A GitHub Actions cron drives ``scripts/kaggle_scheduler.py``, which uses a
private Hugging Face **dataset repo** as a file-backed task queue / state store,
launches the unified Kaggle notebook with a per-run ``runtime_config.json``, and
passes the HF model token through a private Kaggle dataset. See the repo README's
"Kaggle run scheduler" section for the full design and operating guide.
"""

from ac_zero.scheduler.controller import SchedulerConfig, TickReport, run_tick
from ac_zero.scheduler.kaggle import KaggleClient
from ac_zero.scheduler.store import StateStore

__all__ = ["KaggleClient", "SchedulerConfig", "StateStore", "TickReport", "run_tick"]
