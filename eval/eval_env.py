"""Eval-only env wrapper.

Implements the small set of overrides the eval blueprint needs without
touching ``environment.py``:

- ``spawn_mode``: ``"fixed"`` / ``"random"`` / ``"all_rooms"``
- ``max_steps``: monkey-patches the module constant for the duration of an
  episode (the env reads it as a module-level name at every step).
- ``sensor_noise_sd`` / ``drift_alpha``: same trick.
- exposes ground-truth fields that the agent's observation hides, for metric
  collection (real dirt, real battery, real GP arrays).
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

import environment as env_mod
from environment import (
    BLOCKED_CELL_VALUES,
    CURRICULUM_PHASE_STEPS,
    FREE,
    GRID_H,
    GRID_W,
    CleaningEnv,
)

if TYPE_CHECKING:
    from .eval_config import EvalConfig


# Region row/col slices used by ``spawn_mode="all_rooms"``. Chosen to match
# the environment.py furniture layout (Bedroom A: rows 1-7, cols 1-9;
# Bedroom B: rows 1-7, cols 11-18; Living: rows 9-18, cols 1-18).
ROOM_REGIONS: dict[str, tuple[slice, slice]] = {
    "bedroom_a": (slice(1, 8),  slice(1, 10)),
    "bedroom_b": (slice(1, 8),  slice(11, 19)),
    "living":    (slice(9, 19), slice(1, 19)),
}


@dataclass(frozen=True)
class _EnvOverrides:
    max_steps: int | None = None
    sensor_noise_sd: float | None = None
    drift_alpha: float | None = None


@contextlib.contextmanager
def patch_env_constants(overrides: _EnvOverrides):
    """Temporarily override module-level constants the env reads at runtime."""
    saved: dict[str, object] = {}
    if overrides.max_steps is not None:
        saved["MAX_STEPS"] = env_mod.MAX_STEPS
        env_mod.MAX_STEPS = overrides.max_steps
    if overrides.sensor_noise_sd is not None:
        saved["SENSOR_NOISE_SD"] = env_mod.SENSOR_NOISE_SD
        env_mod.SENSOR_NOISE_SD = overrides.sensor_noise_sd
    if overrides.drift_alpha is not None:
        saved["DRIFT_ALPHA"] = env_mod.DRIFT_ALPHA
        env_mod.DRIFT_ALPHA = overrides.drift_alpha
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(env_mod, k, v)


def list_passable_cells(obs_map: np.ndarray) -> list[tuple[int, int]]:
    """All (r, c) that aren't blocked. Charger included."""
    coords = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if obs_map[r, c] not in BLOCKED_CELL_VALUES:
                coords.append((r, c))
    return coords


def list_free_cells_in_region(
    obs_map: np.ndarray, region: tuple[slice, slice]
) -> list[tuple[int, int]]:
    rs, cs = region
    coords = []
    for r in range(*rs.indices(GRID_H)):
        for c in range(*cs.indices(GRID_W)):
            if obs_map[r, c] == FREE:
                coords.append((r, c))
    return coords


def force_spawn(env: CleaningEnv, mode: str, *, region: str | None = None,
                rng: random.Random | None = None) -> tuple[int, int]:
    """Set ``env.global_step`` and (if necessary) override agent position
    after reset. Returns the (r, c) actually used.

    Must be called *after* ``env.reset(seed=...)`` because reset overwrites
    pos_r/pos_c via the env's own ``_sample_start_pos``.
    """
    rng = rng or random
    if mode == "fixed":
        # Reset's curriculum picks (17, 1). Nothing to do.
        return (env.pos_r, env.pos_c)
    if mode == "random":
        # Reset picked uniformly from FREE cells (curriculum phase 2).
        return (env.pos_r, env.pos_c)
    if mode == "all_rooms":
        if region is None:
            raise ValueError("all_rooms spawn requires a region")
        free = list_free_cells_in_region(env.obs_map, ROOM_REGIONS[region])
        if not free:
            raise RuntimeError(f"No free cells in region {region!r}")
        new_r, new_c = rng.choice(free)
        env.pos_r, env.pos_c = int(new_r), int(new_c)
        return (env.pos_r, env.pos_c)
    raise ValueError(f"Unknown spawn_mode: {mode!r}")


def configure_pre_reset(env: CleaningEnv, spawn_mode: str) -> None:
    """Set ``global_step`` so reset's curriculum branches the way we want."""
    if spawn_mode == "fixed" or spawn_mode == "all_rooms":
        env.global_step = 0
    elif spawn_mode == "random":
        env.global_step = CURRICULUM_PHASE_STEPS
    else:
        raise ValueError(f"Unknown spawn_mode: {spawn_mode!r}")


class EvalEnv(gym.Wrapper):
    """CleaningEnv wrapper with eval-only knobs.

    - Forces curriculum phase based on ``cfg.spawn_mode`` (no global_step
      coupling to the training counter).
    - For ``spawn_mode="all_rooms"``, overrides the agent's start position
      to a random cell inside the given room region and rebuilds the obs.
    - Injects per-step ground-truth values into the ``info`` dict under
      ``gt_*`` keys so the metric collector can read them without the agent
      ever seeing them in its observation.

    The env's reward, dynamics, and termination logic are unchanged.
    ``cfg`` overrides (sensor_noise_sd, drift_alpha, max_steps) are handled
    externally by the ``patch_env_constants`` context manager in the runner.
    """

    def __init__(self, base_env: CleaningEnv, cfg: EvalConfig) -> None:
        super().__init__(base_env)
        self._cfg = cfg

    def __getattr__(self, name: str):
        # Gymnasium ≥1.0 removed the __getattr__ forwarder from gym.Wrapper.
        # Re-implement it so runner/metrics code can access CleaningEnv attrs
        # (obs_map, Dirt_Map, pos_r, pos_c, E, t, gp_mu, gp_sigma,
        # _compute_clean_ratio, …) directly through the EvalEnv instance.
        # Block only true dunder names to avoid interfering with Python internals.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # Guard: self.env may not be set yet during super().__init__
        try:
            inner = object.__getattribute__(self, "env")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(inner, name)

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset with spawn-mode forcing.

        ``options`` keys (all optional):
            region (str | None): required for ``spawn_mode="all_rooms"``
            rng (random.Random | None): used for all_rooms cell selection
        """
        opts = options or {}
        configure_pre_reset(self.env, self._cfg.spawn_mode)
        obs, info = self.env.reset(seed=seed)

        if self._cfg.spawn_mode == "all_rooms":
            region: str | None = opts.get("region")
            rng: random.Random | None = opts.get("rng")
            force_spawn(self.env, "all_rooms", region=region, rng=rng)
            # Rebuild obs to reflect the overridden position and current GP belief.
            obs = env_mod._build_obs(
                self.env.obs_map,
                self.env.Dirt_Map,
                self.env.pos_r,
                self.env.pos_c,
                self.env.E,
                mu_gp=float(self.env.gp_mu[self.env.pos_r, self.env.pos_c]),
                sigma_gp=float(self.env.gp_sigma[self.env.pos_r, self.env.pos_c]),
            )

        return obs, info

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the env and inject ground-truth into info (agent never sees it)."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ground-truth fields — hidden from the agent's observation.
        info["gt_t_global"] = int(self.env.t)
        info["gt_mu_gp_local"] = float(
            self.env.gp_mu[self.env.pos_r, self.env.pos_c]
        )
        info["gt_sigma_gp_local"] = float(
            self.env.gp_sigma[self.env.pos_r, self.env.pos_c]
        )
        info["gt_dirt_remaining"] = float(self.env.Dirt_Map.sum())

        return obs, reward, terminated, truncated, info
