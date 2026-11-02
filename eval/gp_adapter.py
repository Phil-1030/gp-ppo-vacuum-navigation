"""GP belief adapters — the compatibility layer between RL-only and live-GP.

The runner only ever sees the ``GPAdapter`` protocol. Today both reference
implementations return constant arrays; tomorrow ``LiveGPAdapter`` wraps the
real GP module without any runner changes.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from environment import GP_STUB_MEAN, GP_STUB_VAR, GRID_H, GRID_W


@runtime_checkable
class GPAdapter(Protocol):
    name: str

    def __call__(
        self,
        obs_map: np.ndarray,
        dirt_map: np.ndarray,
        agent_rc: tuple[int, int],
        t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def reset(self, seed: int) -> None:
        ...


class StubAdapter:
    """Constant μ, σ — bit-for-bit equivalent to env's RL-only stub path."""

    name = "stub"

    def __call__(self, obs_map, dirt_map, agent_rc, t):
        mu = np.full((GRID_H, GRID_W), GP_STUB_MEAN, dtype=np.float32)
        sigma = np.full((GRID_H, GRID_W), GP_STUB_VAR, dtype=np.float32)
        return mu, sigma

    def reset(self, seed: int) -> None:
        return None


class ConstantLiveAdapter:
    """Drop-in 'live' adapter that returns the same constants the stub does.

    Used by the ``gp_compatibility`` suite to verify that switching to a live
    adapter doesn't perturb behaviour. When the real GP module lands, this
    class is replaced by ``LiveGPAdapter(real_gp_module)``.
    """

    name = "live:constant"

    def __call__(self, obs_map, dirt_map, agent_rc, t):
        mu = np.full((GRID_H, GRID_W), GP_STUB_MEAN, dtype=np.float32)
        sigma = np.full((GRID_H, GRID_W), GP_STUB_VAR, dtype=np.float32)
        return mu, sigma

    def reset(self, seed: int) -> None:
        return None


class LiveGPAdapter:
    """Wraps ``IndependentBayesianGrid`` to satisfy the ``GPAdapter`` protocol.

    On each call the adapter:
      1. Observes the agent's current cell with the true (noiseless) dirt value
         from the environment's Dirt_Map.
      2. Runs a Kalman update (+ optional spatial blur).
      3. Returns the current (mu_map, sigma_map) posterior.

    ``reset(seed)`` resets the GP prior for the new episode.  The ``seed``
    argument is accepted for interface compatibility but the GP itself is
    deterministic given its config — the RNG in the training callback is what
    adds noise during training; eval uses noiseless observations here.
    """

    def __init__(self) -> None:
        from environment import CleaningEnv
        from gp.gp_config import GPConfig
        from gp.grid_gp import IndependentBayesianGrid

        self._cfg = GPConfig()
        self._gp = IndependentBayesianGrid(CleaningEnv().obs_map, self._cfg)
        self.name = "live:gp.grid_gp:IndependentBayesianGrid"

    def __call__(
        self,
        obs_map: np.ndarray,
        dirt_map: np.ndarray,
        agent_rc: tuple[int, int],
        t: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        r, c = agent_rc
        d_true = float(dirt_map[r, c])
        self._gp.observe(r, c, d_true)
        self._gp.update()
        return self._gp.posterior()

    def reset(self, seed: int) -> None:
        self._gp.reset()


def build_adapter(spec: str, module_arg: str | None = None) -> GPAdapter:
    """Resolve an adapter spec string into a live adapter instance.

    spec format:
        "stub"
        "live:<dotted.module.path>:<ClassName>"
    """
    if spec == "stub":
        return StubAdapter()
    if spec.startswith("live:"):
        _, dotted, cls_name = spec.split(":", 2)
        mod = importlib.import_module(dotted)
        cls = getattr(mod, cls_name)
        if module_arg is not None:
            return cls(module_arg)
        return cls()
    raise ValueError(f"Unknown gp_adapter spec: {spec!r}")
