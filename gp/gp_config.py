"""Frozen configuration dataclass for the GP belief module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from environment import GP_STUB_MEAN, GP_STUB_VAR, SENSOR_NOISE_SD


@dataclass(frozen=True)
class GPConfig:
    # ── Prior ────────────────────────────────────────────────────────────────
    # Must match the env stubs (GP_STUB_MEAN, GP_STUB_VAR) so that a freshly
    # reset GP and the StubAdapter produce bit-identical obs[55:57] values
    # before the agent has visited any cell.
    mu_prior:  float = GP_STUB_MEAN    # 0.50
    var_prior: float = GP_STUB_VAR     # 1.00  (σ_prior = 1.0)

    # ── Observation model ────────────────────────────────────────────────────
    # Matches env's SENSOR_NOISE_SD so the Kalman gain is consistent with the
    # noise level the agent's local dirt observation already experiences.
    noise_var: float = SENSOR_NOISE_SD ** 2   # 0.15² = 0.0225

    # ── Spatial blur ─────────────────────────────────────────────────────────
    # After per-cell Kalman updates, a weighted Gaussian blur propagates
    # certainty from visited cells to their unvisited neighbours.
    use_spatial_blur:     bool  = True
    blur_length_scale:    float = 1.5   # Gaussian kernel σ in grid cells
    blur_variance_decay:  float = 0.85  # var multiplier applied to blurred cells

    # ── Training update schedule ─────────────────────────────────────────────
    # Number of env-steps per worker between posterior recomputes and
    # env.update_gp_belief() injections.  1 = every step (eval default).
    gp_update_every: int = 5

    # ── Episode scoping ──────────────────────────────────────────────────────
    # Dirt regenerates between episodes, so the GP resets with the env.
    reset_on_episode_end: bool = True

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_json(self, path: Path) -> None:
        """Write config as JSON (saved alongside PPOConfig's config.json)."""
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: Path) -> "GPConfig":
        data = json.loads(path.read_text())
        return cls(**data)

    def with_overrides(self, **kwargs) -> "GPConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)
