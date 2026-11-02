"""CleaningEnv — RL environment for a 20x20 cleaning robot world.

Implementation of the blueprint specified in CLAUDE.md (v1.0). The GP/ML
integration is stubbed: GP belief inputs are present in the observation
interface but held at fixed values until the ML team replaces them.
"""

from __future__ import annotations

import random
from collections import deque
from math import pi
from typing import Any

import gymnasium as gym
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Global Constants
# ─────────────────────────────────────────────────────────────────────────────

# Grid
GRID_H = 20
GRID_W = 20

# Cell type codes (Obstacle_Map values)
FREE = 0
WALL = 1
CHARGER = 2
TABLE = 3
CABINET = 4
BED = 5

# Convenience tuples used by every check that distinguishes "can the agent
# stand here?" from "is this a navigation obstacle?". Adding a new obstacle
# class only requires updating these tuples.
BLOCKED_CELL_VALUES = (WALL, TABLE, CABINET, BED)
PASSABLE_CELL_VALUES = (FREE, CHARGER)

# Dirt model
N_HOTSPOTS = 12
DRIFT_ALPHA = 0.30
DRIFT_OMEGA = 2 * pi / 300
REGEN_DELAY = 500
SENSOR_NOISE_SD = 0.15

# Action space
N_ACTIONS = 4
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),   # Up
    1: (+1, 0),   # Down
    2: (0, -1),   # Left
    3: (0, +1),   # Right
}

# Battery
BATTERY_DRAIN_PER_STEP = 0.002   # 【修改】从 0.005 改为 0.002。满电现在能活 500 步，完美对齐 MAX_STEPS！
CHARGE_RATE_PER_STEP = 0.050

R_CLEAN_SCALE = 15.0       # 提高基础清洁分
R_MOVE = -0.10
R_COLLISION = -1.00
R_DEAD = -500.0
R_SUCCESS = 2000.0         # 【关键】从 +50 提到 +2000，通关才是终极目标
CHARGE_REWARD_SCALE = 5.0  # 稍微降低充电即时奖励，防止它赖在桩上不走

# 废除面包屑，引入清洁衰减
CLEAN_DECAY_FACTOR = 0.5   # 【修改】从 0.5 改为 2.0。加大引路力度，如果不听话扣分会非常痛。

# Intrinsic reward
BETA_INIT = 0.50
BETA_MIN = 0.05
BETA_DECAY = 0.0005

# Episode management
MAX_STEPS = 1500
SUCCESS_THRESHOLD = 0.05

# GP stub values
GP_STUB_MEAN = 0.50
GP_STUB_VAR = 1.00

# Curriculum
CURRICULUM_PHASE_STEPS = 50_000

# Charger location (used in scalar observation). v1.1: relocated from the
# bedroom corner (1, 1) into the living room corner (18, 1).
CHARGER_POS = np.array([18, 1], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Grid & Obstacle Map
# ─────────────────────────────────────────────────────────────────────────────

FURNITURE: list[tuple[int, slice, slice]] = [
    # (class, row_slice, col_slice)
    (BED,     slice(1, 6),   slice(7, 10)),    # BA  — bedroom A bed (5 × 3)
    (BED,     slice(1, 6),   slice(11, 14)),   # BB  — bedroom B bed (5 × 3)
    (CABINET, slice(1, 2),   slice(3, 7)),     # KA  — bedroom A wardrobe (1 × 4)
    (CABINET, slice(1, 2),   slice(14, 17)),   # KB  — bedroom B wardrobe (1 × 3)
    (TABLE,   slice(6, 8),   slice(1, 3)),     # TA  — bedroom A side table (2 × 2)
    (TABLE,   slice(6, 8),   slice(17, 19)),   # TB  — bedroom B side table (2 × 2)
    (TABLE,   slice(11, 13), slice(6, 14)),    # TL1 — living-room dining table (2 × 8)
    (TABLE,   slice(15, 17), slice(9, 11)),    # TL2 — living-room coffee table (2 × 2)
    (CABINET, slice(13, 17), slice(1, 2)),     # KL1 — living-room left bookshelf (4 × 1)
    (CABINET, slice(11, 15), slice(18, 19)),   # KL2 — living-room right bookshelf (4 × 1)
    (CABINET, slice(18, 19), slice(7, 13)),    # KL3 — TV / sofa unit (1 × 6)
    (CABINET, slice(18, 19), slice(15, 18)),   # KL4 — bottom-wall storage (1 × 3)
]


def _build_obstacle_map() -> np.ndarray:
    obs = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    # Outer boundary walls
    obs[0, :] = WALL
    obs[19, :] = WALL
    obs[:, 0] = WALL
    obs[:, 19] = WALL

    # Bedroom A / Bedroom B vertical separator (rows 1–7 only).
    obs[1:8, 10] = WALL

    # Bedrooms / Living room horizontal separator at row 8.
    obs[8, 1:19] = WALL

    # Carve Doorway A ↔ Living room
    obs[8, 4:6] = FREE

    # Carve Doorway B ↔ Living room
    obs[8, 14:16] = FREE

    # Stamp furniture before the charger so the charger cell can never be
    # overwritten by a furniture slice.
    for cls, rs, cs in FURNITURE:
        obs[rs, cs] = cls

    # Charging station — bottom-left corner of the living room.
    obs[18, 1] = CHARGER

    return obs


def _verify_connectivity(obs: np.ndarray) -> bool:
    start = (18, 1)  # charger
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < GRID_H
                and 0 <= nc < GRID_W
                and obs[nr, nc] not in BLOCKED_CELL_VALUES
            ):
                if (nr, nc) not in visited:
                    queue.append((nr, nc))
    passable_cells = int(np.isin(obs, PASSABLE_CELL_VALUES).sum())
    assert len(visited) == passable_cells, (
        f"Connectivity check failed: {len(visited)} reachable, "
        f"{passable_cells} total passable cells."
    )
    return True


# Run topology check at import time.
_verify_connectivity(_build_obstacle_map())


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Dirt Model
# ─────────────────────────────────────────────────────────────────────────────

# Furniture-derived dirt hotspots. Order, sigma, weight and phi must stay in
# sync with HOTSPOT_PHASES used by _build_phase_map below.
HOTSPOTS: list[dict] = [
    {"center": (6.5,  1.5),  "sigma": 1.5, "weight": 0.5, "phi": 0.0},          # TA
    {"center": (6.5,  17.5), "sigma": 1.5, "weight": 0.5, "phi": 0.0},          # TB
    {"center": (11.5, 9.5),  "sigma": 4.0, "weight": 1.0, "phi": pi},           # TL1
    {"center": (15.5, 9.5),  "sigma": 2.0, "weight": 0.7, "phi": pi},           # TL2
    {"center": (3.0,  8.0),  "sigma": 2.5, "weight": 0.7, "phi": 3 * pi / 2},   # BA
    {"center": (3.0,  12.0), "sigma": 2.5, "weight": 0.7, "phi": 3 * pi / 2},   # BB
    {"center": (1.0,  4.5),  "sigma": 1.5, "weight": 0.4, "phi": pi / 4},       # KA
    {"center": (1.0,  15.0), "sigma": 1.5, "weight": 0.4, "phi": pi / 4},       # KB
    {"center": (14.5, 1.0),  "sigma": 1.5, "weight": 0.4, "phi": 3 * pi / 4},   # KL1
    {"center": (12.5, 18.0), "sigma": 1.5, "weight": 0.4, "phi": 3 * pi / 4},   # KL2
    {"center": (18.0, 9.5),  "sigma": 2.0, "weight": 0.5, "phi": 3 * pi / 4},   # KL3
    {"center": (18.0, 16.0), "sigma": 1.5, "weight": 0.4, "phi": 3 * pi / 4},   # KL4
]


def _build_D_base(obs: np.ndarray) -> np.ndarray:
    rows, cols = np.mgrid[0:GRID_H, 0:GRID_W]
    D = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    for h in HOTSPOTS:
        r0, c0 = h["center"]
        dist_sq = (rows - r0) ** 2 + (cols - c0) ** 2
        D += h["weight"] * np.exp(-dist_sq / (2 * h["sigma"] ** 2))

    # Mask all blocked cells (walls + furniture) before normalising so the peak
    # is computed only over passable cells.
    blocked = np.isin(obs, BLOCKED_CELL_VALUES)
    D[blocked] = 0.0

    peak = D[~blocked].max()
    if peak > 0:
        D /= peak

    return D.astype(np.float32)


def _build_phase_map(obs: np.ndarray) -> np.ndarray:
    centers = np.array([h["center"] for h in HOTSPOTS], dtype=np.float32)
    phases = np.array([h["phi"] for h in HOTSPOTS], dtype=np.float32)

    rows, cols = np.mgrid[0:GRID_H, 0:GRID_W]
    dists = np.stack(
        [(rows - cr) ** 2 + (cols - cc) ** 2 for cr, cc in centers],
        axis=0,
    )
    nearest = np.argmin(dists, axis=0)

    phase_map = phases[nearest].astype(np.float32)
    phase_map[np.isin(obs, BLOCKED_CELL_VALUES)] = 0.0
    return phase_map


def _update_dirt(
    Dirt_Map: np.ndarray,
    D_base: np.ndarray,
    regen_cooldown: np.ndarray,
    phase_map: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    regen_cooldown = np.maximum(0, regen_cooldown - 1)

    drift_factor = 1.0 + DRIFT_ALPHA * np.sin(DRIFT_OMEGA * t + phase_map)
    D_t_full = np.clip(D_base * drift_factor, 0.0, 1.0).astype(np.float32)

    active_mask = (regen_cooldown == 0)
    Dirt_Map = np.where(active_mask, D_t_full, Dirt_Map).astype(np.float32)

    return Dirt_Map, regen_cooldown


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Observation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_obs(
    obs_map: np.ndarray,
    dirt_map: np.ndarray,
    r: int,
    c: int,
) -> tuple[np.ndarray, np.ndarray]:
    pad = 2
    padded_obs = np.pad(obs_map, pad, mode="constant", constant_values=WALL)
    padded_dirt = np.pad(dirt_map, pad, mode="constant", constant_values=0.0)

    obs_local = padded_obs[r:r + 5, c:c + 5]
    dirt_local = padded_dirt[r:r + 5, c:c + 5]

    obs_local = np.isin(obs_local, BLOCKED_CELL_VALUES).astype(np.float32)

    noise = np.random.normal(0, SENSOR_NOISE_SD, size=(5, 5)).astype(np.float32)
    dirt_local_noisy = np.clip(dirt_local + noise, 0.0, 1.0).astype(np.float32)

    return obs_local.flatten(), dirt_local_noisy.flatten()


def _get_scalars(
    r: int,
    c: int,
    E: float,
    mu_gp: float = GP_STUB_MEAN,
    sigma_gp: float = GP_STUB_VAR,
) -> np.ndarray:
    delta = CHARGER_POS - np.array([r, c], dtype=np.float32)
    delta_norm = delta / np.array([GRID_H, GRID_W], dtype=np.float32)
    pos_norm = np.array([r / (GRID_H - 1), c / (GRID_W - 1)], dtype=np.float32)
    return np.array(
        [E, delta_norm[0], delta_norm[1], pos_norm[0], pos_norm[1], mu_gp, sigma_gp],
        dtype=np.float32,
    )


def _build_obs(
    obs_map: np.ndarray,
    dirt_map: np.ndarray,
    r: int,
    c: int,
    E: float,
    mu_gp: float = GP_STUB_MEAN,
    sigma_gp: float = GP_STUB_VAR,
) -> np.ndarray:
    local_obs, local_dirt = _get_local_obs(obs_map, dirt_map, r, c)
    scalars = _get_scalars(r, c, E, mu_gp, sigma_gp)
    obs = np.concatenate([local_obs, local_dirt, scalars]).astype(np.float32)
    assert obs.shape == (57,), f"Observation shape mismatch: {obs.shape}"
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Reward Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intrinsic(E: float, sigma_gp: float, beta: float) -> float:
    alpha_E = max(0.0, (E - 0.30) / 0.70)
    return beta * alpha_E * sigma_gp


def _update_beta(episode_count: int) -> float:
    return max(BETA_MIN, BETA_INIT - episode_count * BETA_DECAY)


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 — CleaningEnv class
# ─────────────────────────────────────────────────────────────────────────────

class CleaningEnv(gym.Env):
    """20x20 cleaning robot environment. See CLAUDE.md for the spec."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=np.inf, shape=(57,), dtype=np.float32
        )

        # Static obstacle map — built once and reused.
        self.obs_map = _build_obstacle_map()
        _verify_connectivity(self.obs_map)

        # GP interface (stubbed during RL-only phase).
        self.gp_ready = False
        self.gp_mu = np.full((GRID_H, GRID_W), GP_STUB_MEAN, dtype=np.float32)
        self.gp_sigma = np.full((GRID_H, GRID_W), GP_STUB_VAR, dtype=np.float32)

        # Counters that persist across episodes.
        self.global_step = 0
        self.episode_count = 0
        self.beta = BETA_INIT

        # Episode-scoped state (populated in reset()).
        self.D_base: np.ndarray = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.phase_map: np.ndarray = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.Dirt_Map: np.ndarray = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.regen_cooldown: np.ndarray = np.zeros((GRID_H, GRID_W), dtype=np.int32)
        self.pos_r: int = 1
        self.pos_c: int = 2
        self.E: float = 1.0
        self.t: int = 0
        self.step_count: int = 0
        self.episode_return: float = 0.0
        self.cells_cleaned: int = 0
        self.clean_count_map = np.zeros((GRID_H, GRID_W), dtype=np.int32)

    def update_gp_belief(self, mu_map: np.ndarray, sigma_map: np.ndarray) -> None:
        """Called by the ML module after each GP posterior update."""
        self.gp_mu = mu_map.astype(np.float32)
        self.gp_sigma = sigma_map.astype(np.float32)
        self.gp_ready = True

    def _compute_clean_ratio(self) -> float:
        passable_mask = np.isin(self.obs_map, PASSABLE_CELL_VALUES)
        current_dirt = float(self.Dirt_Map[passable_mask].sum())
        max_dirt = float(self.D_base[passable_mask].sum())
        if max_dirt <= 0:
            return 1.0
        return 1.0 - (current_dirt / max_dirt)

    def _build_info(
        self,
        v: float = 0.0,
        R_int: float = 0.0,
        on_charger: bool = False,
    ) -> dict[str, Any]:
        return {
            "position": (self.pos_r, self.pos_c),
            "battery": round(self.E, 4),
            "dirt_cleaned": round(v, 4),
            "cells_cleaned": self.cells_cleaned,
            "clean_ratio": round(self._compute_clean_ratio(), 4),
            "R_int": round(R_int, 4),
            "on_charger": on_charger,
            "step": self.step_count,
            "episode_return": round(self.episode_return, 4),
        }

    def _sample_start_pos(self) -> tuple[int, int]:
        if self.global_step < CURRICULUM_PHASE_STEPS:
            # One cell above the charger at (18, 1).
            return (17, 1)
        free_cells = list(zip(*np.where(self.obs_map == FREE)))
        choice = random.choice(free_cells)
        return (int(choice[0]), int(choice[1]))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1. Obstacle map already built in __init__ — reuse self.obs_map.

        # 2. Dirt model
        self.D_base = _build_D_base(self.obs_map)
        self.phase_map = _build_phase_map(self.obs_map)
        self.Dirt_Map = self.D_base.copy()
        self.regen_cooldown = np.zeros((GRID_H, GRID_W), dtype=np.int32)
        self.clean_count_map = np.zeros((GRID_H, GRID_W), dtype=np.int32)
        # 3. Agent state
        self.pos_r, self.pos_c = self._sample_start_pos()
        self.E = 1.0
        self.t = 0
        self.step_count = 0
        self.episode_return = 0.0
        self.cells_cleaned = 0

        # 4. Beta — refresh from current episode count.
        self.episode_count += 1
        self.beta = _update_beta(self.episode_count)

        # 5. Build initial observation
        obs = _build_obs(
            self.obs_map, self.Dirt_Map, self.pos_r, self.pos_c, self.E
        )
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        R = 0.0
        terminated = False
        truncated = False

        # ── Step 1: Compute intended next position ────────────────────────
        dr, dc = ACTION_DELTAS[action]
        next_r = self.pos_r + dr
        next_c = self.pos_c + dc
        
        # [NEW] 记录移动前到充电桩的曼哈顿距离
        charger_r, charger_c = int(CHARGER_POS[0]), int(CHARGER_POS[1])
        old_dist = abs(self.pos_r - charger_r) + abs(self.pos_c - charger_c)

        # ── Step 2: Collision check ───────────────────────────────────────
        out_of_bounds = not (0 <= next_r < GRID_H and 0 <= next_c < GRID_W)
        is_blocked = (not out_of_bounds) and (
            int(self.obs_map[next_r, next_c]) in BLOCKED_CELL_VALUES
        )
        is_collision = out_of_bounds or is_blocked

        # ── Step 3: Position update ───────────────────────────────────────
        if not is_collision:
            self.pos_r, self.pos_c = next_r, next_c
        if is_collision:
            R += R_COLLISION
            
        # [NEW] 记录移动后到充电桩的曼哈顿距离
        new_dist = abs(self.pos_r - charger_r) + abs(self.pos_c - charger_c)

        # ── Step 4: Battery drain ─────────────────────────────────────────
        self.E -= BATTERY_DRAIN_PER_STEP
        self.E = max(0.0, self.E)

        # ── Step 5: Cleaning settlement ──
        v = float(self.Dirt_Map[self.pos_r, self.pos_c])
        
        # 【修改】引入边际递减效应
        if v > 0.01:
            # 基础奖励 * (衰减系数 ^ 之前清理过的次数)
            R += (v * R_CLEAN_SCALE) * (CLEAN_DECAY_FACTOR ** self.clean_count_map[self.pos_r, self.pos_c])
            self.clean_count_map[self.pos_r, self.pos_c] += 1
            self.cells_cleaned += 1
            
        self.Dirt_Map[self.pos_r, self.pos_c] = 0.0
        self.regen_cooldown[self.pos_r, self.pos_c] = REGEN_DELAY

        # ── Step 6: Charging ──────────────────────────────────────────────
        on_charger = (self.obs_map[self.pos_r, self.pos_c] == CHARGER)
        if on_charger:
            charge_amount = min(CHARGE_RATE_PER_STEP, 1.0 - self.E)
            self.E += charge_amount
            R += charge_amount * CHARGE_REWARD_SCALE

        # ── Step 7: Composite reward (step penalty + intrinsic + breadcrumbs) ───────────
        R += R_MOVE

        

        sigma_gp = (
            float(self.gp_sigma[self.pos_r, self.pos_c])
            if self.gp_ready else GP_STUB_VAR
        )
        mu_gp = (
            float(self.gp_mu[self.pos_r, self.pos_c])
            if self.gp_ready else GP_STUB_MEAN
        )
        R_int = _compute_intrinsic(self.E, sigma_gp, self.beta)
        R += R_int

        # ── Step 8: Dirt map update ───────────────────────────────────────
        self.t += 1
        self.Dirt_Map, self.regen_cooldown = _update_dirt(
            self.Dirt_Map, self.D_base, self.regen_cooldown, self.phase_map, self.t
        )

        # ── Step 9: Step counter ──────────────────────────────────────────
        self.step_count += 1
        self.global_step += 1
        self.episode_return += R

        # ── Step 10: Termination checks ───────────────────────────────────
        if self.E <= 0.0 and not on_charger:
            R += R_DEAD
            terminated = True
        elif self._compute_clean_ratio() >= (1.0 - SUCCESS_THRESHOLD):
            R += R_SUCCESS
            terminated = True
        elif self.step_count >= MAX_STEPS:
            truncated = True

        # ── Step 11: Build observation ────────────────────────────────────
        obs = _build_obs(
            self.obs_map, self.Dirt_Map, self.pos_r, self.pos_c,
            self.E, mu_gp, sigma_gp,
        )
        info = self._build_info(v=v, R_int=R_int, on_charger=on_charger)

        return obs, float(R), terminated, truncated, info

    def render(self) -> Any:
        raise NotImplementedError("render() is deferred — see CLAUDE.md Section 13.")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = CleaningEnv()
    obs, info = env.reset()

    # Check 1: observation shape
    assert obs.shape == (57,) and obs.dtype == np.float32

    # Check 2: obs values in legal range (no NaNs, no infs)
    assert np.all(np.isfinite(obs))

    # Check 3: run 200 random steps without exception
    total_reward = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (57,)
        assert np.all(np.isfinite(obs))
        if terminated or truncated:
            obs, info = env.reset()

    # Check 4: battery is always in [0, 1]
    assert 0.0 <= obs[50] <= 1.0

    print("Smoke test passed.")
    print(f"  Total reward over 200 steps: {total_reward:.2f}")
    print(f"  Final battery: {obs[50]:.3f}")
    print(f"  Final clean ratio: {info['clean_ratio']:.3f}")
