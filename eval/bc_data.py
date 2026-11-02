"""Behavior-cloning data collection using BFSGreedyPolicy as the teacher.

Public API
----------
collect_bc_data  — run teacher episodes, return (X, y) numpy arrays
save_bc_data     — persist to compressed .npz
load_bc_data     — load from .npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from environment import MAX_STEPS, CleaningEnv

from .policies import BFSGreedyPolicy


def collect_bc_data(
    n_episodes: int = 200,
    seed_base: int = 9000,
    battery_threshold: float = 0.30,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BFSGreedyPolicy for n_episodes and collect (obs, action) pairs.

    Returns
    -------
    X : np.ndarray, shape (N, 57), float32
        Raw 57-dim observation vectors recorded before each action.
    y : np.ndarray, shape (N,), int32
        Actions taken (0–3).
    """
    effective_max_steps = max_steps if max_steps is not None else MAX_STEPS

    policy = BFSGreedyPolicy(battery_threshold=battery_threshold)
    env = CleaningEnv()

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []

    for ep_idx in range(n_episodes):
        seed = seed_base + ep_idx
        policy.reset(seed)
        obs, _ = env.reset(seed=seed)

        step = 0
        while step < effective_max_steps:
            action = policy.act(obs, env)
            all_obs.append(obs.astype(np.float32))
            all_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            if terminated or truncated:
                break

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Collected episode {ep_idx + 1}/{n_episodes} "
                  f"(total samples so far: {len(all_obs):,})")

    env.close()

    X = np.stack(all_obs, axis=0)          # (N, 57) float32
    y = np.array(all_actions, dtype=np.int32)  # (N,) int32
    return X, y


def save_bc_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save (X, y) to a compressed .npz file. Keys: 'X', 'y'."""
    np.savez_compressed(path, X=X, y=y)


def load_bc_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from a .npz file produced by save_bc_data."""
    data = np.load(path)
    return data["X"], data["y"]
