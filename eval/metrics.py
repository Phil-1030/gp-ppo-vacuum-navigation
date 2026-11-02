"""Per-step and per-episode metric collection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from environment import (
    GRID_H,
    GRID_W,
    N_ACTIONS,
    PASSABLE_CELL_VALUES,
    R_DEAD,
    R_SUCCESS,
    SUCCESS_THRESHOLD,
    CleaningEnv,
)


@dataclass
class StepRecord:
    step: int
    t_global: int           # env.t after this step (drift-model clock)
    pos_r: int
    pos_c: int
    action: int
    is_collision: bool
    dirt_collected: float
    battery: float
    on_charger: bool
    R_total: float
    R_intrinsic: float
    R_extrinsic: float
    mu_gp_local: float      # GP mean at agent position (gt, injected by EvalEnv)
    sigma_gp_local: float   # GP sigma at agent position
    dirt_remaining_total: float  # sum of Dirt_Map after this step


@dataclass
class EpisodeMetrics:
    seed: int
    policy: str
    spawn_mode: str
    region: str | None
    gp_adapter_name: str

    # Aggregate fields populated as the episode runs.
    ep_length: int = 0
    ep_return_total: float = 0.0
    ep_return_intrinsic: float = 0.0
    ep_return_extrinsic: float = 0.0
    n_collisions: int = 0
    n_charger_visits: int = 0
    total_charge_steps: int = 0
    total_dirt_collected: float = 0.0
    cells_cleaned_unique: int = 0
    first_charger_visit_step: float = float("nan")
    sum_battery: float = 0.0
    min_battery: float = float("inf")
    time_below_30pct: int = 0

    # Filled at finalize().
    clean_ratio_final: float = 0.0
    terminated_cause: str = ""
    visit_counts: np.ndarray = field(default_factory=lambda: np.zeros((GRID_H, GRID_W), dtype=np.int32))
    action_counts: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.int32))
    n_revisits: int = 0
    mean_revisit_count: float = 0.0

    # Optional per-step trace (used for trajectory plots and capture-steps mode).
    step_records: list[StepRecord] = field(default_factory=list)

    # Final-state GP snapshots for diagnostic plots (populated in finalize_metrics).
    gp_mu_final: np.ndarray | None = field(default=None, repr=False)
    gp_sigma_final: np.ndarray | None = field(default=None, repr=False)
    dirt_map_final: np.ndarray | None = field(default=None, repr=False)

    # Time-integrated cleaning and coverage metrics (filled at finalize()).
    maintenance_ratio: float = 0.0
    peak_clean_ratio: float = 0.0
    coverage_rate: float = 0.0

    # Internal: track on-charger transitions and per-step clean-ratio history.
    _was_on_charger: bool = field(default=False, repr=False)
    _clean_ratio_history: list[float] = field(default_factory=list, repr=False)


def update_metrics(
    m: EpisodeMetrics,
    *,
    env: CleaningEnv,
    info: dict,
    reward: float,
    action: int,
    prev_pos: tuple[int, int],
    keep_step_record: bool,
) -> None:
    """Apply one env step's effects to an EpisodeMetrics object."""
    pos = (env.pos_r, env.pos_c)
    is_collision = (pos == prev_pos)
    dirt = float(info.get("dirt_cleaned", 0.0))
    battery = float(info.get("battery", env.E))
    on_charger = bool(info.get("on_charger", False))
    R_int = float(info.get("R_int", 0.0))
    R_ext = float(reward) - R_int

    m.ep_length += 1
    m.ep_return_total += float(reward)
    m.ep_return_intrinsic += R_int
    m.ep_return_extrinsic += R_ext
    m.total_dirt_collected += dirt
    m.sum_battery += battery
    if battery < m.min_battery:
        m.min_battery = battery
    if battery < 0.30:
        m.time_below_30pct += 1
    if is_collision:
        m.n_collisions += 1
    if on_charger:
        m.total_charge_steps += 1
        if not m._was_on_charger:
            m.n_charger_visits += 1
            if math.isnan(m.first_charger_visit_step):
                m.first_charger_visit_step = float(m.ep_length)
    m._was_on_charger = on_charger
    m._clean_ratio_history.append(float(info.get("clean_ratio", 0.0)))

    m.visit_counts[pos[0], pos[1]] += 1
    m.action_counts[action] += 1

    if keep_step_record:
        m.step_records.append(
            StepRecord(
                step=m.ep_length,
                t_global=int(info.get("gt_t_global", 0)),
                pos_r=pos[0], pos_c=pos[1],
                action=action,
                is_collision=is_collision,
                dirt_collected=dirt,
                battery=battery,
                on_charger=on_charger,
                R_total=float(reward),
                R_intrinsic=R_int,
                R_extrinsic=R_ext,
                mu_gp_local=float(info.get("gt_mu_gp_local", 0.0)),
                sigma_gp_local=float(info.get("gt_sigma_gp_local", 0.0)),
                dirt_remaining_total=float(info.get("gt_dirt_remaining", 0.0)),
            )
        )


def finalize_metrics(
    m: EpisodeMetrics,
    *,
    env: CleaningEnv,
    info: dict,
    terminated: bool,
    truncated: bool,
    final_reward: float,
) -> EpisodeMetrics:
    m.cells_cleaned_unique = int(info.get("cells_cleaned", 0))
    m.clean_ratio_final = float(info.get("clean_ratio", env._compute_clean_ratio()))

    if terminated:
        if m.clean_ratio_final >= (1.0 - SUCCESS_THRESHOLD):
            m.terminated_cause = "SUCCESS"
        elif env.E <= 0.0 + 1e-9:
            m.terminated_cause = "DEAD"
        else:
            m.terminated_cause = "TERMINATED_OTHER"
    elif truncated:
        m.terminated_cause = "TRUNCATED"
    else:
        m.terminated_cause = "INCOMPLETE"

    visited_mask = m.visit_counts > 0
    visited = m.visit_counts[visited_mask]
    revisits = visited[visited > 1]
    m.n_revisits = int(revisits.size)
    m.mean_revisit_count = float(revisits.mean()) if revisits.size > 0 else 0.0

    if m._clean_ratio_history:
        m.maintenance_ratio = float(np.mean(m._clean_ratio_history))
        m.peak_clean_ratio = float(max(m._clean_ratio_history))
    passable_cells = int(np.isin(env.obs_map, PASSABLE_CELL_VALUES).sum())
    unique_visited = int(visited_mask.sum())
    m.coverage_rate = unique_visited / passable_cells if passable_cells > 0 else 0.0

    # Capture final GP belief and ground-truth dirt for diagnostic plots.
    # Works with both CleaningEnv directly and EvalEnv (gym.Wrapper forwards attrs).
    m.gp_mu_final = np.array(env.gp_mu, dtype=np.float32)
    m.gp_sigma_final = np.array(env.gp_sigma, dtype=np.float32)
    m.dirt_map_final = np.array(env.Dirt_Map, dtype=np.float32)

    return m


def to_csv_row(m: EpisodeMetrics) -> dict:
    """Flat dict suitable for CSV writing."""
    return {
        "seed": m.seed,
        "policy": m.policy,
        "spawn_mode": m.spawn_mode,
        "region": m.region or "",
        "gp_adapter_name": m.gp_adapter_name,
        "terminated_cause": m.terminated_cause,
        "ep_length": m.ep_length,
        "ep_return_total": round(m.ep_return_total, 4),
        "ep_return_extrinsic": round(m.ep_return_extrinsic, 4),
        "ep_return_intrinsic": round(m.ep_return_intrinsic, 4),
        "clean_ratio_final": round(m.clean_ratio_final, 4),
        "cells_cleaned_unique": m.cells_cleaned_unique,
        "total_dirt_collected": round(m.total_dirt_collected, 4),
        "n_collisions": m.n_collisions,
        "n_charger_visits": m.n_charger_visits,
        "total_charge_steps": m.total_charge_steps,
        "mean_battery": round(m.sum_battery / max(1, m.ep_length), 4),
        "min_battery": round(m.min_battery, 4) if m.min_battery != float("inf") else float("nan"),
        "time_below_30pct": m.time_below_30pct,
        "first_charger_visit_step": m.first_charger_visit_step,
        "n_revisits": m.n_revisits,
        "mean_revisit_count": round(m.mean_revisit_count, 4),
        "maintenance_ratio": round(m.maintenance_ratio, 4),
        "peak_clean_ratio": round(m.peak_clean_ratio, 4),
        "coverage_rate": round(m.coverage_rate, 4),
    }


CSV_FIELDS = (
    "seed", "policy", "spawn_mode", "region", "gp_adapter_name",
    "terminated_cause", "ep_length",
    "ep_return_total", "ep_return_extrinsic", "ep_return_intrinsic",
    "clean_ratio_final", "cells_cleaned_unique", "total_dirt_collected",
    "n_collisions", "n_charger_visits", "total_charge_steps",
    "mean_battery", "min_battery", "time_below_30pct",
    "first_charger_visit_step", "n_revisits", "mean_revisit_count",
    "maintenance_ratio", "peak_clean_ratio", "coverage_rate",
)


def expected_terminal_bonus(cause: str) -> float:
    """Reward bonus that the env added on the terminal step."""
    if cause == "SUCCESS":
        return R_SUCCESS
    if cause == "DEAD":
        return R_DEAD
    return 0.0
