"""Derive a stable checkpoint name from a training configuration.

Two runs that train the *same model on the same task* should share a checkpoint
name, so their best models chain into one warm-start lineage on Hugging Face
(``model_checkpoints/<name>/``). The name is built from the fields that define
that task and model, split into two parts:

* a readable slug -- ``rank``, ``agent``, ``model``, ``moveset``, ``reward_mode``
* a short hash of the remaining task-defining fields (goal, discount, relator
  bound, difficulty cap) so configs that differ only there never collide silently

Operational knobs that do not change *what* the model learns (iteration count,
worker count, learning rate, batch size, run directory, seed, local dataset
paths) are deliberately excluded, so re-running with a bigger time budget or a
different learning rate still resolves to the same lineage.
"""

from __future__ import annotations

import hashlib
import json

from ac_zero.training.pipeline.pipeline_config import TrainingPipelineConfig

# Fields that appear verbatim in the readable part of the name (after ``rank``).
_READABLE_FIELDS = ("agent", "model", "moveset", "reward_mode")
# Task-defining fields folded into the trailing hash: they change what the model
# learns to do but are not worth spelling out in a human-readable slug.
_HASHED_FIELDS = (
    "goal_mode",
    "goal_reward",
    "gamma",
    "max_relator_tokens",
    "scramble_depth",
    "dataset_max_difficulty",
)


def _slug(value: object) -> str:
    """Normalize a field value to a filesystem/URL-safe token."""
    return str(value).lower().replace("-", "_").replace("/", "_").replace(" ", "_")


def derive_checkpoint_name(config: TrainingPipelineConfig) -> str:
    """Return the deterministic checkpoint name for ``config``.

    Example: ``rank2-ppo-residual_mlp-strict_ac-length_reduction_and_goal-1a2b3c``.
    The same task/model configuration always yields the same name; a change to
    any hashed task field yields a new one.

    ``checkpoint_name_suffix`` splits one such task into parallel lineages that
    are otherwise identical -- the pretraining ablation's ``pretrained``/``scratch``
    twins, whose configs differ only in a field the name does not read. It lands
    in the readable part, before the hash, so the hash stays the trailing element
    and *both* lineages still fork together when a hashed task field changes.
    Pinning a literal ``checkpoint_name`` instead would give the twins distinct
    names but silently freeze them across such a change.
    """
    readable = "-".join(
        [
            f"rank{config.rank}",
            *(_slug(getattr(config, field)) for field in _READABLE_FIELDS),
            *([_slug(config.checkpoint_name_suffix)] if config.checkpoint_name_suffix else []),
        ]
    )
    payload = {field: getattr(config, field) for field in _HASHED_FIELDS}
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:6]
    return f"{readable}-{digest}"
