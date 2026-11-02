"""IndependentBayesianGrid — per-cell Kalman GP with optional RBF spatial blur.

Algorithm (one episode):
  reset()             → all cells restored to prior: mu=0.5, var=1.0
  observe(r, c, d)    → buffer one noisy dirt measurement
  update()            → Kalman update each buffered cell; optional spatial blur
  posterior()         → (mu_map, sigma_map) as float32 20×20 arrays

Computational cost (blur_length_scale=1.5, gp_update_every=5):
  observe()    O(1)        ~100 ns per call
  update()     O(P + K²P)  ~250 µs for P≈240 passable cells, K=11 kernel
  posterior()  O(P)        ~20 µs
  8 workers × every 5 steps ≈ 500 µs total per training step (<3% overhead)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from environment import BLOCKED_CELL_VALUES, GRID_H, GRID_W

if TYPE_CHECKING:
    from .gp_config import GPConfig


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy convolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_kernel(sigma: float) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel with radius = ceil(3σ) cells."""
    half = max(1, int(np.ceil(3.0 * sigma)))
    ksize = 2 * half + 1          # always odd
    y, x = np.mgrid[-half: half + 1, -half: half + 1]
    k = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    return k.astype(np.float64)


def _conv2d_reflect(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D correlation of ``arr`` with ``kernel`` using reflect-padding.

    Pure NumPy; no scipy.  Works for any odd-sized kernel on any 2-D array.
    Reflect padding avoids border artefacts (edge cell values mirror inward).
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = arr.shape
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i: i + H, j: j + W]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class IndependentBayesianGrid:
    """Per-cell Gaussian belief model over dirt levels on the 20×20 grid.

    Each passable cell independently maintains a Gaussian posterior
    (mu, var) over its dirt level.  When visited, the posterior is
    updated via a single-step Kalman filter (Gaussian–Gaussian conjugate
    update), which is exact and O(1) per observation:

        k          = var / (var + noise_var)          # Kalman gain ∈ (0, 1)
        mu        += k × (d_obs − mu)                 # pull toward observation
        var       *= (1 − k)                           # reduce uncertainty

    After applying all buffered observations, an optional weighted Gaussian
    blur propagates certainty from observed to unvisited neighbouring cells
    (implemented without scipy via manual 2-D kernel convolution).

    Blocked cells (walls, furniture) are masked throughout: they carry
    zero certainty and are excluded from all updates.

    Episode scoping: call reset() at the start of each episode.  Dirt
    regenerates between episodes, so carrying stale posteriors forward
    would mislead the agent.
    """

    def __init__(self, obs_map: np.ndarray, cfg: "GPConfig") -> None:
        self._cfg = cfg

        # Boolean mask: True where the agent can never stand.
        self._blocked: np.ndarray = np.isin(obs_map, BLOCKED_CELL_VALUES)
        self._passable: np.ndarray = ~self._blocked

        # Precompute the Gaussian blur kernel once.
        self._kernel = _build_kernel(cfg.blur_length_scale)

        # Posterior arrays — float64 internally, cast to float32 on output.
        self._mu  = np.empty((GRID_H, GRID_W), dtype=np.float64)
        self._var = np.empty((GRID_H, GRID_W), dtype=np.float64)

        # Observation buffer: list of (row, col, noisy_dirt_value).
        self._buffer: list[tuple[int, int, float]] = []

        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore the uniform prior. Call at the start of every episode."""
        self._mu[:] = self._cfg.mu_prior
        self._var[:] = self._cfg.var_prior
        # Blocked cells: zero certainty so they never contribute to blurs.
        self._mu[self._blocked]  = 0.0
        self._var[self._blocked] = 0.0
        self._buffer.clear()

    def observe(self, r: int, c: int, d_noisy: float) -> None:
        """Buffer one noisy dirt observation at cell (r, c).

        Does not trigger a posterior update.  Call update() to flush.
        Safe to call multiple times before update() — all observations
        in the buffer are applied in order during the next update().
        """
        self._buffer.append((r, c, float(d_noisy)))

    def update(self) -> None:
        """Apply all buffered observations and optionally blur the posterior.

        Steps:
          1. Kalman update for each (r, c, d_noisy) in the buffer.
          2. If use_spatial_blur: propagate certainty to unvisited neighbours.
          3. Clear the buffer.
        """
        for r, c, d in self._buffer:
            if self._blocked[r, c]:
                continue   # safety: skip if somehow a blocked cell slipped in
            v = self._var[r, c]
            k_gain = v / (v + self._cfg.noise_var)
            self._mu[r, c]  += k_gain * (d - self._mu[r, c])
            self._var[r, c] *= (1.0 - k_gain)
        self._buffer.clear()

        if self._cfg.use_spatial_blur:
            self._apply_spatial_blur()

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu_map, sigma_map) as float32 20×20 arrays.

        sigma_map = sqrt(var_map), consistent with env.gp_sigma convention.
        Blocked cells are zeroed out — the agent never stands there.
        Safe to call without a preceding update(); returns the most recently
        computed posterior.
        """
        mu_out = np.where(self._passable, self._mu,  0.0).astype(np.float32)
        sg_out = np.where(self._passable, np.sqrt(self._var), 0.0).astype(np.float32)
        return mu_out, sg_out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_spatial_blur(self) -> None:
        """Weighted Gaussian blur: visited cells inform unvisited neighbours.

        Uses a certainty-weighted average so that well-observed cells
        (high certainty = low var) dominate over uncertain ones.
        Only cells whose posterior variance is still above half the prior
        (i.e. never or rarely visited) are updated by the blur — observed
        cells keep their precise Kalman posterior.

        Implementation notes (pure NumPy, no scipy):
          - Certainty = 1 / var for passable cells, 0 for blocked.
          - blurred_mu = conv(mu × certainty) / conv(certainty)
          - Apply only to high-uncertainty passable cells.
          - var is reduced by blur_variance_decay in the update zone.
          - var is floored at noise_var so sigma never collapses below
            the measurement noise level (~0.15).
          - mu is clamped to [0, 1] (valid dirt range).
        """
        # Certainty map: blocked cells have zero certainty.
        certainty = np.zeros((GRID_H, GRID_W), dtype=np.float64)
        certainty[self._passable] = 1.0 / (self._var[self._passable] + 1e-8)

        # Weighted blur of mu and of the certainty normaliser.
        blurred_num  = _conv2d_reflect(self._mu * certainty, self._kernel)
        blurred_denom = _conv2d_reflect(certainty,            self._kernel)
        blurred_mu    = blurred_num / (blurred_denom + 1e-8)

        # Update only high-uncertainty passable cells.
        high_unc = self._passable & (self._var > 0.5 * self._cfg.var_prior)
        self._mu[high_unc]   = blurred_mu[high_unc]
        self._var[high_unc] *= self._cfg.blur_variance_decay

        # Enforce floors and clamps on passable cells.
        p = self._passable
        self._var[p] = np.maximum(self._var[p], self._cfg.noise_var)
        self._mu[p]  = np.clip(self._mu[p], 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from environment import CleaningEnv, GP_STUB_MEAN, GP_STUB_VAR
    from gp.gp_config import GPConfig

    obs_map = CleaningEnv().obs_map
    cfg = GPConfig()
    gp = IndependentBayesianGrid(obs_map, cfg)

    # ── Check 1: prior matches stub ──────────────────────────────────────────
    mu0, sg0 = gp.posterior()
    passable = ~np.isin(obs_map, BLOCKED_CELL_VALUES)
    assert np.allclose(mu0[passable], GP_STUB_MEAN, atol=1e-6), \
        "Prior mu should match GP_STUB_MEAN"
    assert np.allclose(sg0[passable], GP_STUB_VAR ** 0.5, atol=1e-6), \
        "Prior sigma should match sqrt(GP_STUB_VAR)"
    # (GP_STUB_VAR = 1.0, so sigma_prior = 1.0 — matches stub)
    print("Check 1 passed: prior matches stub values.")

    # ── Check 2: single observation reduces sigma at visited cell ────────────
    gp.reset()
    gp.observe(17, 1, 0.6)   # charger cell neighbour
    gp.update()
    mu1, sg1 = gp.posterior()
    assert sg1[17, 1] < sg0[17, 1], \
        "Sigma should decrease after observation"
    assert sg1[17, 1] < 0.5, \
        "Sigma should be well below prior after one observation"
    print(f"Check 2 passed: sigma[17,1] = {sg1[17,1]:.4f} (was {sg0[17,1]:.4f}).")

    # ── Check 3: spatial blur propagates to unvisited neighbours ────────────
    # Passable neighbours of (17, 1): (17, 2) is FREE; (16, 1) is CABINET
    # (blocked) so we must test the free neighbour (17, 2).
    neighbour_sigma_before = float(sg0[17, 2])
    neighbour_sigma_after  = float(sg1[17, 2])
    assert neighbour_sigma_after < neighbour_sigma_before, \
        "Spatial blur should reduce sigma in unvisited neighbours"
    print(f"Check 3 passed: neighbour sigma[17,2] = {neighbour_sigma_after:.4f}"
          f" (was {neighbour_sigma_before:.4f}).")

    # ── Check 4: repeated observations converge to noise floor ──────────────
    gp.reset()
    for _ in range(20):
        gp.observe(10, 5, 0.5 + 0.1 * (np.random.rand() - 0.5))
        gp.update()
    _, sg20 = gp.posterior()
    noise_floor = cfg.noise_var ** 0.5
    assert sg20[10, 5] <= noise_floor * 1.1, \
        f"After 20 observations sigma should approach noise floor {noise_floor:.4f}"
    print(f"Check 4 passed: sigma[10,5] after 20 obs = {sg20[10,5]:.4f}"
          f" (noise floor = {noise_floor:.4f}).")

    # ── Check 5: blocked cells stay zero ─────────────────────────────────────
    blocked_cells = np.argwhere(np.isin(obs_map, BLOCKED_CELL_VALUES))
    r_b, c_b = blocked_cells[0]
    mu_b, sg_b = gp.posterior()
    assert mu_b[r_b, c_b] == 0.0 and sg_b[r_b, c_b] == 0.0, \
        "Blocked cells must be zero in posterior"
    print("Check 5 passed: blocked cells are zero in posterior.")

    # ── Check 6: reset restores prior ────────────────────────────────────────
    gp.reset()
    mu_r, sg_r = gp.posterior()
    assert np.allclose(mu_r[passable], GP_STUB_MEAN, atol=1e-6), \
        "reset() must restore mu to prior"
    assert np.allclose(sg_r[passable], 1.0, atol=1e-6), \
        "reset() must restore sigma to 1.0"
    print("Check 6 passed: reset() restores prior exactly.")

    # ── Check 7: posterior arrays are float32 ────────────────────────────────
    mu_f, sg_f = gp.posterior()
    assert mu_f.dtype == np.float32 and sg_f.dtype == np.float32, \
        "posterior() must return float32 arrays"
    assert mu_f.shape == (GRID_H, GRID_W) and sg_f.shape == (GRID_H, GRID_W), \
        f"posterior() must return {GRID_H}×{GRID_W} arrays"
    print("Check 7 passed: posterior() returns float32 20×20 arrays.")

    # ── Check 8: convolution kernel sums to 1.0 ──────────────────────────────
    k = _build_kernel(cfg.blur_length_scale)
    assert abs(k.sum() - 1.0) < 1e-10, "Kernel must be normalised"
    print(f"Check 8 passed: kernel shape {k.shape}, sum = {k.sum():.10f}.")

    # ── Performance check ────────────────────────────────────────────────────
    import time
    gp.reset()
    for r, c in [(17, 1), (16, 1), (15, 1), (17, 2), (16, 2)]:
        gp.observe(r, c, 0.4)
    t0 = time.perf_counter()
    for _ in range(1000):
        gp.update()
        gp.posterior()
        # Refill buffer for the next iteration
        gp.observe(17, 1, 0.4)
    elapsed_us = (time.perf_counter() - t0) * 1e3  # ms for 1000 calls
    per_call_us = elapsed_us                        # ms total → µs per call (×1000 / 1000)
    print(f"Performance: {elapsed_us:.1f} ms / 1000 update+posterior calls"
          f"  ({elapsed_us:.2f} µs per call).")

    print("\nAll checks passed.")
