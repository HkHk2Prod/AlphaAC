from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ac_zero.agents.greedy import GreedySolver
from ac_zero.certificates.verifier import CertificateVerifier
from ac_zero.datasets.generator import generate_solvable
from ac_zero.encoding.padded import StateEncoder
from ac_zero.environment.env import ACEnvironment, ACEnvironmentConfig
from ac_zero.models.registry import create_model
from ac_zero.search.mcts import UniformMCTS
from ac_zero.system.manifests import ReproducibilityManifest
from ac_zero.training.callbacks import CallbackManager, default_smoke_callbacks
from ac_zero.training.checkpointing import CheckpointManager
from ac_zero.training.losses import return_to_go
from ac_zero.training.replay_buffer import ReplayBuffer


@dataclass(frozen=True, slots=True)
class SmokeTrainingSummary:
    """High-level result of the complete CPU smoke training workflow."""

    run_directory: str
    certificate_path: str
    checkpoint_path: str
    benchmark_ready: bool
    model_name: str
    mcts_simulations: int
    replay_size: int
    optimizer_updates: int
    checkpoint_restored: bool
    certificate_verified: bool
    event_log_path: str
    progress_log_path: str
    live_graph_path: str
    final_graph_path: str


def run_smoke_training(
    seed: int,
    run_directory: str | Path = "runs/smoke",
    callbacks: CallbackManager | None = None,
) -> SmokeTrainingSummary:
    """Exercise the full lightweight training stack on a one-move instance.

    The smoke workflow is intentionally tiny, but it is not a no-op: it encodes
    a state, applies every registered smoke model name, runs a root MCTS search,
    records replay data, computes return-to-go targets, performs a deterministic
    scalar optimizer update, saves and reloads a checkpoint, solves a fixture,
    and independently verifies the emitted certificate.
    """

    run = Path(run_directory)
    checkpoint_dir = run / "checkpoints"
    certificate_dir = run / "certificates"
    artifact_dir = run / "artifacts"
    log_dir = run / "logs"
    for directory in (checkpoint_dir, certificate_dir, artifact_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manager = callbacks or default_smoke_callbacks(run)

    rng = random.Random(seed)
    try:
        manager.emit(0, "start", "starting smoke training", {"seed": seed})
        instance = generate_solvable(rank=2, depth=1, seed=seed)
        env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=4))
        manager.emit(
            1,
            "dataset",
            "generated one solvable presentation",
            {
                "initial_length": instance.presentation.total_length,
                "rank": instance.presentation.rank,
            },
        )

        encoder = StateEncoder(max_word_length=8)
        encoding = encoder.encode(env.state)
        action_count = len(env.catalog)
        mask = env.legal_action_mask()
        manager.emit(
            2,
            "encoding",
            "encoded current Markov state",
            {"action_count": action_count, "legal_actions": sum(1 for ok in mask if ok)},
        )

        model_outputs = _evaluate_smoke_models(encoding, action_count)
        for offset, (name, output) in enumerate(model_outputs.items(), start=3):
            manager.emit(
                offset,
                "model",
                f"evaluated {name}",
                {
                    "value": float(output["value"]),
                    "logits_finite": bool(output["logits_finite"]),
                },
            )

        mcts = UniformMCTS(simulations=8)
        stats = mcts.search(env)
        manager.emit(
            8,
            "mcts",
            "ran root search",
            {
                "mcts_simulations": mcts.simulations,
                "root_expanded_nodes": stats.expanded_nodes,
            },
        )

        replay = ReplayBuffer[dict[str, Any]](capacity=16)
        replay.add(
            {
                "tokens_shape": list(encoding.tokens.shape),
                "action_mask": list(mask),
                "visit_counts": list(stats.visit_counts),
                "chosen_action": mcts.select_action(env),
                "model_version": "smoke-v1",
            }
        )
        sampled_batch = replay.sample(1, rng)
        manager.emit(9, "replay", "recorded and sampled replay item", {"replay_size": len(replay)})

        rewards = _rollout_rewards(instance.reverse_moves, instance.presentation)
        targets = return_to_go([float(reward) for reward in rewards])
        target = targets[0] if targets else 0.0
        optimizer_state = _one_scalar_optimizer_update(
            prediction=model_outputs["residual_mlp"]["value"],
            target=target,
        )
        manager.emit(
            10,
            "optimizer",
            "completed one scalar optimizer update",
            {
                "loss": float(optimizer_state["loss"]),
                "target": target,
                "gradient": float(optimizer_state["gradient"]),
                "optimizer_updates": int(optimizer_state["step"]),
            },
        )

        checkpoint_manager = CheckpointManager(checkpoint_dir)
        checkpoint_path = checkpoint_manager.save_json(
            "latest",
            {
                "schema_version": "aczero-smoke-checkpoint-v1",
                "seed": seed,
                "model_outputs": model_outputs,
                "optimizer_state": optimizer_state,
                "replay_size": len(replay),
                "sampled_batch_size": len(sampled_batch),
                "mcts": asdict(stats),
            },
        )
        restored = checkpoint_manager.load_json("latest")
        checkpoint_restored = restored["optimizer_state"] == optimizer_state
        manager.emit(
            11,
            "checkpoint",
            "saved and restored checkpoint",
            {"checkpoint_restored": checkpoint_restored},
        )

        certificate_path = certificate_dir / "example.json"
        solve_env = ACEnvironment(instance.presentation, ACEnvironmentConfig(max_moves=4))
        result = GreedySolver().solve(
            solve_env,
            certificate_path=certificate_path,
            experiment_id="smoke",
            seed=seed,
        )
        certificate_verified = bool(
            result.success and CertificateVerifier().verify_path(certificate_path).ok
        )
        manager.emit(
            12,
            "certificate",
            "solved fixture and verified certificate",
            {
                "certificate_verified": certificate_verified,
                "best_reduction": result.best_reduction,
            },
        )

        metrics = {
            "seed": seed,
            "model_forward_passes": len(model_outputs),
            "mcts_simulations": mcts.simulations,
            "root_expanded_nodes": stats.expanded_nodes,
            "replay_size": len(replay),
            "optimizer_updates": optimizer_state["step"],
            "checkpoint_restored": checkpoint_restored,
            "certificate_verified": certificate_verified,
            "return_to_go": targets,
        }
        (run / "metrics.jsonl").write_text(json.dumps(metrics, sort_keys=True) + "\n")
        ReproducibilityManifest.create(
            "smoke", seed, {"rank": 2, "depth": 1, "models": list(model_outputs)}
        ).write(run / "manifest.json")

        summary = SmokeTrainingSummary(
            run_directory=str(run),
            certificate_path=str(certificate_path),
            checkpoint_path=str(checkpoint_path),
            benchmark_ready=True,
            model_name="residual_mlp",
            mcts_simulations=mcts.simulations,
            replay_size=len(replay),
            optimizer_updates=int(optimizer_state["step"]),
            checkpoint_restored=checkpoint_restored,
            certificate_verified=certificate_verified,
            event_log_path=str(log_dir / "training_events.jsonl"),
            progress_log_path=str(log_dir / "progress.log"),
            live_graph_path=str(artifact_dir / "live_graphs.txt"),
            final_graph_path=str(artifact_dir / "final_graphs.txt"),
        )
        manager.emit(
            13,
            "completed",
            "smoke training completed",
            {
                "optimizer_updates": summary.optimizer_updates,
                "certificate_verified": summary.certificate_verified,
            },
        )
        (artifact_dir / "smoke_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
        )
        return summary
    except Exception as exc:
        manager.emit_error("error", "smoke training failed", exc)
        raise
    finally:
        manager.close()


def _evaluate_smoke_models(encoding: Any, action_count: int) -> dict[str, dict[str, Any]]:
    """Run all registered architecture names through their smoke forward pass."""

    outputs: dict[str, dict[str, Any]] = {}
    for name in ("uniform", "residual_mlp", "gru", "transformer", "deepsets"):
        output = create_model(name).apply(encoding, action_count)
        outputs[name] = {
            "logits_shape": list(output.logits.shape),
            "logits_finite": bool(np.isfinite(output.logits).all()),
            "value": float(output.value),
            "value_finite": bool(np.isfinite(output.value)),
        }
    return outputs


def _rollout_rewards(moves: tuple[Any, ...], initial: Any) -> list[int]:
    """Replay a known fixture path and compute canonical best-length rewards."""

    pres = initial
    best = initial.total_length
    rewards: list[int] = []
    for move in moves:
        nxt = move.apply(pres)
        next_best = min(best, nxt.total_length)
        rewards.append(best - next_best)
        best = next_best
        pres = nxt
    return rewards or [0]


def _one_scalar_optimizer_update(prediction: float, target: float) -> dict[str, float | int]:
    """Perform one deterministic scalar gradient-descent step for smoke metrics."""

    parameter = 0.0
    learning_rate = 0.1
    gradient = 2.0 * (prediction + parameter - target)
    parameter -= learning_rate * gradient
    loss = (prediction + parameter - target) ** 2
    return {
        "step": 1,
        "learning_rate": learning_rate,
        "gradient": gradient,
        "parameter": parameter,
        "loss": loss,
    }
