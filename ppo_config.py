"""Frozen hyperparameter set for the PPO training pipeline.

The single source of truth for every knob exposed by ``train_ppo.py``.
Every run dumps an instance of ``PPOConfig`` to ``runs/<run_name>/config.json``
so the exact values used can always be reconstructed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    # ── Run identity ─────────────────────────────────────────────────────
    run_name: str | None = None
    seed: int = 42

    # ── Vectorisation ────────────────────────────────────────────────────
    n_envs: int = 8

    # ── Schedule ─────────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10

    # ── Optimiser ────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Discount / GAE ───────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── Network ──────────────────────────────────────────────────────────
    policy: str = "MlpPolicy"
    net_arch: tuple[int, ...] = (128, 128)

    # ── Logging / checkpointing ──────────────────────────────────────────
    checkpoint_freq: int = 50_000
    eval_freq: int = 25_000
    eval_episodes: int = 10
    log_dir: str = "runs"

    # ── Plot rolling-window ──────────────────────────────────────────────
    plot_window: int = 50

    # ── GP integration ───────────────────────────────────────────────────
    gp_run: bool = False

    # ── Resume ───────────────────────────────────────────────────────────
    resume_from: str | None = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"ppo_{ts}"

    def to_json(self, path: Path) -> None:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def with_overrides(self, **kwargs) -> "PPOConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if "net_arch" in clean and isinstance(clean["net_arch"], list):
            clean["net_arch"] = tuple(clean["net_arch"])
        return replace(self, **clean)
