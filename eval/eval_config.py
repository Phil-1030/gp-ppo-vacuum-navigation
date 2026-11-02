"""Frozen EvalConfig dataclass + six pre-defined suite factories.

The blueprint mandates that everything that can change between experiments is
captured in this single object so the experiment is reproducible from
``eval_config.json`` alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class EvalConfig:
    # Identity
    experiment_name: str
    model_path: str
    run_root: str = "runs/ppo_v1"

    # Episode budget
    n_episodes: int = 100
    seed_base: int = 1000
    max_steps: int | None = None

    # Spawn condition: "fixed" | "random" | "all_rooms"
    spawn_mode: str = "fixed"

    # Action selection
    deterministic: bool = True

    # GP compatibility hook
    gp_adapter: str = "stub"          # "stub" | "live:<dotted.path>:<ClassName>"
    gp_module_arg: str | None = None  # passed to live adapter constructor
    gp_update_every: int = 1

    # Stochasticity overrides (None → env defaults)
    sensor_noise_sd: float | None = None
    drift_alpha: float | None = None

    # Logging
    capture_steps: bool = False
    representative_episodes: int = 3

    # Comparison policies
    baselines: tuple[str, ...] = ("random", "greedy_dirt")

    def to_json(self, path: Path) -> None:
        data = asdict(self)
        data["baselines"] = list(self.baselines)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def with_overrides(self, **kwargs) -> "EvalConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if "baselines" in clean and isinstance(clean["baselines"], list):
            clean["baselines"] = tuple(clean["baselines"])
        return replace(self, **clean)


# ─────────────────────────────────────────────────────────────────────────────
# Suite factories — each takes (model_path, run_root) and returns EvalConfig.
# ─────────────────────────────────────────────────────────────────────────────

SuiteFactory = Callable[[str, str], EvalConfig]


def _baseline(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="baseline",
        model_path=model_path,
        run_root=run_root,
        n_episodes=100,
        spawn_mode="fixed",
        baselines=("random", "greedy_dirt"),
    )


def _generalization(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="generalization",
        model_path=model_path,
        run_root=run_root,
        n_episodes=200,
        spawn_mode="random",
        baselines=("random", "greedy_dirt"),
    )


def _room_breakdown(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="room_breakdown",
        model_path=model_path,
        run_root=run_root,
        n_episodes=60,
        spawn_mode="all_rooms",
        baselines=("random",),
    )


def _long_horizon(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="long_horizon",
        model_path=model_path,
        run_root=run_root,
        n_episodes=100,
        spawn_mode="fixed",
        max_steps=2000,
        baselines=("random",),
    )


def _noise_robustness(model_path: str, run_root: str) -> EvalConfig:
    # Single-noise-level eval; the CLI is responsible for sweeping if asked.
    # The default invocation tests the trained sensor noise (0.15).
    return EvalConfig(
        experiment_name="noise_robustness",
        model_path=model_path,
        run_root=run_root,
        n_episodes=50,
        spawn_mode="fixed",
        sensor_noise_sd=0.15,
        baselines=("random",),
    )


def _gp_compatibility(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="gp_compatibility",
        model_path=model_path,
        run_root=run_root,
        n_episodes=50,
        spawn_mode="fixed",
        gp_adapter="live:eval.gp_adapter:ConstantLiveAdapter",
        baselines=("random",),
    )


def _gp_exploration(model_path: str, run_root: str) -> EvalConfig:
    """Live GP suite: uses IndependentBayesianGrid and adds gp_greedy baseline.

    Tests whether the GP-trained agent (ppo_gp_v1) achieves a higher
    clean_ratio than the stub-trained baseline, and whether gp_greedy
    outperforms random exploration when given a live sigma map.
    """
    return EvalConfig(
        experiment_name="gp_exploration",
        model_path=model_path,
        run_root=run_root,
        n_episodes=100,
        spawn_mode="random",
        gp_adapter="live:eval.gp_adapter:LiveGPAdapter",
        baselines=("random", "greedy_dirt", "gp_greedy"),
        capture_steps=True,
        representative_episodes=5,
    )


def _bc_baseline(model_path: str, run_root: str) -> EvalConfig:
    return EvalConfig(
        experiment_name="bc_baseline",
        model_path=model_path,
        run_root=run_root,
        n_episodes=100,
        spawn_mode="fixed",
        baselines=(
            "random",
            "greedy_dirt",
            "lr:runs/bc/lr_model.pkl",
            "svm:runs/bc/svm_model.pkl",
        ),
    )


SUITES: dict[str, SuiteFactory] = {
    "baseline":         _baseline,
    "generalization":   _generalization,
    "room_breakdown":   _room_breakdown,
    "long_horizon":     _long_horizon,
    "noise_robustness": _noise_robustness,
    "gp_compatibility": _gp_compatibility,
    "gp_exploration":   _gp_exploration,
    "bc_baseline":      _bc_baseline,
}
