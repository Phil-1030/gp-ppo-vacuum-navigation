"""GPTrainingCallback — injects GP beliefs into the VecEnv during PPO training.

One IndependentBayesianGrid per DummyVecEnv worker.  Each worker maintains an
independent episode-scoped posterior that is updated from the env's ``info``
dict and injected back via ``env_method('update_gp_belief', ...)``.

Update schedule (per worker):
  Every step     → observe(r, c, d_noisy)   buffered, O(1)
  Every K steps  → update() + posterior() + env_method inject
  On episode end → gp.reset(), counter reset

This keeps training overhead under ~700 µs / step across all 8 workers.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gp.gp_config import GPConfig
from gp.grid_gp import IndependentBayesianGrid


class GPTrainingCallback(BaseCallback):
    """SB3 callback that maintains per-worker GP beliefs during training.

    Parameters
    ----------
    obs_map:
        The static 20×20 obstacle map (``CleaningEnv().obs_map``).  Built once
        outside and shared across all GP instances (read-only).
    gp_cfg:
        Frozen configuration.  ``gp_cfg.gp_update_every`` controls how often
        the posterior is recomputed and injected.
    verbose:
        SB3 verbosity level (0 = silent).
    """

    def __init__(
        self,
        obs_map: np.ndarray,
        gp_cfg: GPConfig,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._obs_map = obs_map
        self._cfg = gp_cfg

        # Built in _on_training_start once n_envs is known.
        self._gps: list[IndependentBayesianGrid] = []
        self._step_counters: np.ndarray = np.zeros(0, dtype=np.int64)
        self._rng: np.random.Generator = np.random.default_rng(42)

    # ── SB3 lifecycle hooks ───────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._gps = [
            IndependentBayesianGrid(self._obs_map, self._cfg)
            for _ in range(n)
        ]
        self._step_counters = np.zeros(n, dtype=np.int64)

        # Seed noise RNG from the model's own seed for reproducibility.
        seed = int(getattr(self.model, "seed", None) or 42)
        self._rng = np.random.default_rng(seed)

        if self.verbose:
            print(
                f"[GPTrainingCallback] initialised {n} GP workers, "
                f"gp_update_every={self._cfg.gp_update_every}, "
                f"spatial_blur={self._cfg.use_spatial_blur}"
            )

    def _on_step(self) -> bool:
        """Called once per env-step (across all n_envs workers simultaneously).

        Flow for each worker i:
          1. Read (r, c) and noiseless dirt from info.
          2. Add simulated sensor noise → buffer observation.
          3. If not done AND periodic: update posterior + inject into env.
          4. If done: reset GP for the next episode.

        Separating the done-branch from the periodic-update branch ensures we
        never push a stale old-episode posterior into the freshly reset env.
        """
        infos: list[dict] = self.locals["infos"]
        dones: np.ndarray  = self.locals["dones"]

        noise_sd = float(self._cfg.noise_var ** 0.5)

        for i, (info, done) in enumerate(zip(infos, dones)):
            # ── 1. Observe ────────────────────────────────────────────────
            r, c = info["position"]
            d_true = float(info.get("dirt_cleaned", 0.0))
            # Simulated sensor noise — matches env's SENSOR_NOISE_SD
            d_noisy = float(np.clip(
                d_true + self._rng.normal(0.0, noise_sd),
                0.0, 1.0,
            ))
            self._gps[i].observe(r, c, d_noisy)
            self._step_counters[i] += 1

            if done:
                # ── 4. Episode end: reset GP for the next episode ─────────
                # Do NOT inject: the env has already auto-reset; pushing an
                # old-episode posterior would pollute the new episode's belief.
                if self._cfg.reset_on_episode_end:
                    self._gps[i].reset()
                self._step_counters[i] = 0

            elif self._step_counters[i] % self._cfg.gp_update_every == 0:
                # ── 3. Periodic posterior update + injection ──────────────
                self._gps[i].update()
                mu, sigma = self._gps[i].posterior()
                # env_method propagates through VecMonitor → DummyVecEnv →
                # _ExposeInfoForMonitor.__getattr__ → CleaningEnv
                self.training_env.env_method(
                    "update_gp_belief", mu, sigma, indices=[i]
                )

        return True  # returning False would stop training

    def _on_training_end(self) -> None:
        if self.verbose:
            print("[GPTrainingCallback] training ended; GP workers released.")
