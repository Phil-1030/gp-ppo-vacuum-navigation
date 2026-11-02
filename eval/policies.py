"""Policy adapters with a uniform interface.

The runner only ever calls ``policy.reset(seed)`` and ``policy.act(obs, env)``,
so the trained PPO model and the analytical baselines are interchangeable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from stable_baselines3 import PPO

from environment import (
    ACTION_DELTAS,
    BLOCKED_CELL_VALUES,
    CHARGER_POS,
    GRID_H,
    GRID_W,
    N_ACTIONS,
    CleaningEnv,
)


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...
    def act(self, obs: np.ndarray, env: CleaningEnv) -> int: ...


class PPOPolicy:
    def __init__(self, path: str, deterministic: bool = True):
        self.name = f"ppo:{Path(path).stem}"
        self._model = PPO.load(path, device="cpu")
        self._det = deterministic

    def reset(self, seed: int) -> None:
        self._model.set_random_seed(seed)

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        action, _ = self._model.predict(obs, deterministic=self._det)
        return int(action)


class RandomPolicy:
    name = "random"

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        assert self._rng is not None, "RandomPolicy.reset() not called"
        return int(self._rng.integers(0, N_ACTIONS))


class GreedyDirtPolicy:
    """Reference baseline. Cheats: reads the env's internal ``Dirt_Map``.

    From the agent's current cell, considers the four neighbours; picks the
    passable neighbour with maximum dirt. Ties broken at random. If every
    neighbour is blocked, picks a uniform random action (will collide).
    """

    name = "greedy_dirt"

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        assert self._rng is not None, "GreedyDirtPolicy.reset() not called"
        candidates: list[tuple[float, int]] = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = env.pos_r + dr, env.pos_c + dc
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if int(env.obs_map[nr, nc]) in BLOCKED_CELL_VALUES:
                continue
            candidates.append((float(env.Dirt_Map[nr, nc]), action))
        if not candidates:
            return int(self._rng.integers(0, N_ACTIONS))
        max_dirt = max(c[0] for c in candidates)
        best = [a for d, a in candidates if d >= max_dirt - 1e-9]
        return int(self._rng.choice(best))


class GPGreedyPolicy:
    """Information-greedy baseline: moves toward the in-sight neighbour with
    the highest σ_gp value (maximum epistemic uncertainty).

    This policy requires a live GP adapter that produces non-trivial σ values.
    In stub mode every cell has the same σ, making the choice degenerate.
    The body is intentionally left unimplemented in v1.0 — fill it in once the
    real GP module is available and ``gp_sigma`` carries meaningful gradients.
    """

    name = "gp_greedy"

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        assert self._rng is not None, "GPGreedyPolicy.reset() not called"
        candidates: list[tuple[float, int]] = []
        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = env.pos_r + dr, env.pos_c + dc
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if int(env.obs_map[nr, nc]) in BLOCKED_CELL_VALUES:
                continue
            candidates.append((float(env.gp_sigma[nr, nc]), action))
        if not candidates:
            return int(self._rng.integers(0, N_ACTIONS))
        max_sigma = max(c[0] for c in candidates)
        best = [a for s, a in candidates if s >= max_sigma - 1e-9]
        return int(self._rng.choice(best))


class BFSGreedyPolicy:
    """Teacher for behavior cloning.

    Cheats: reads env.Dirt_Map and env.E directly (acceptable for a teacher).
    Finds nearest dirty cell via BFS; returns to charger when battery < threshold.
    Ties broken by (row, col) lexicographic order for reproducibility.
    """

    name = "bfs_greedy"

    def __init__(self, battery_threshold: float = 0.30) -> None:
        self._battery_threshold = battery_threshold

    def reset(self, seed: int) -> None:
        pass  # stateless

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        charger_r, charger_c = int(CHARGER_POS[0]), int(CHARGER_POS[1])

        if env.E < self._battery_threshold:
            goal: tuple[int, int] = (charger_r, charger_c)
        else:
            dirty = self._bfs_nearest_dirty(env)
            goal = dirty if dirty is not None else (charger_r, charger_c)

        if (env.pos_r, env.pos_c) == goal:
            return int(np.random.randint(0, N_ACTIONS))

        action = self._bfs_first_step(env, goal)
        return action if action is not None else int(np.random.randint(0, N_ACTIONS))

    def _bfs_nearest_dirty(self, env: CleaningEnv) -> tuple[int, int] | None:
        start = (env.pos_r, env.pos_c)
        visited: set[tuple[int, int]] = {start}
        queue: list[tuple[int, int]] = [start]
        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            if env.Dirt_Map[r, c] > 0.01:
                return (r, c)
            neighbors: list[tuple[int, int]] = []
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if int(env.obs_map[nr, nc]) in BLOCKED_CELL_VALUES:
                    continue
                if (nr, nc) in visited:
                    continue
                neighbors.append((nr, nc))
            for nr, nc in sorted(neighbors):
                visited.add((nr, nc))
                queue.append((nr, nc))
        return None

    def _bfs_first_step(self, env: CleaningEnv, goal: tuple[int, int]) -> int | None:
        start = (env.pos_r, env.pos_c)
        # Maps cell -> (parent_cell, action_taken_to_reach_cell)
        parent_map: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        queue: list[tuple[int, int]] = [start]
        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            if (r, c) == goal:
                break
            neighbors: list[tuple[int, int, int]] = []
            for action, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if int(env.obs_map[nr, nc]) in BLOCKED_CELL_VALUES:
                    continue
                if (nr, nc) in parent_map:
                    continue
                neighbors.append((nr, nc, action))
            for nr, nc, action in sorted(neighbors, key=lambda x: (x[0], x[1])):
                parent_map[(nr, nc)] = ((r, c), action)
                queue.append((nr, nc))

        if goal not in parent_map:
            return None

        cell = goal
        while True:
            entry = parent_map[cell]
            if entry is None:
                return None
            parent, action = entry
            if parent == start:
                return action
            cell = parent


class BCPolicy:
    """Runs a joblib-serialized sklearn classifier (LR or LinearSVC) as a Policy.

    Only uses the obs vector — no cheating on env internals. Clean BC inference.
    """

    def __init__(self, model_path: str, label: str) -> None:
        self.name = f"{label}:{Path(model_path).stem}"
        self._clf = joblib.load(model_path)

    def reset(self, seed: int) -> None:
        pass  # stateless

    def act(self, obs: np.ndarray, env: CleaningEnv) -> int:
        return int(self._clf.predict(obs.reshape(1, -1))[0])


def build_policy(spec: str, deterministic: bool = True) -> Policy:
    """Resolve a policy spec into an instance.

    spec format:
        "random"
        "greedy_dirt"
        "gp_greedy"
        "<path/to/model.zip>"   (anything else)
    """
    if spec.startswith("lr:"):
        return BCPolicy(spec[3:], "lr")
    if spec.startswith("svm:"):
        return BCPolicy(spec[4:], "svm")
    if spec == "random":
        return RandomPolicy()
    if spec == "greedy_dirt":
        return GreedyDirtPolicy()
    if spec == "gp_greedy":
        return GPGreedyPolicy()
    return PPOPolicy(spec, deterministic=deterministic)
