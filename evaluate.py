"""Evaluation harness CLI for trained CleaningEnv PPO models.

Examples:
    python evaluate.py --suite baseline
    python evaluate.py --suite generalization --n-episodes 500
    python evaluate.py --model runs/ppo_v1/final_model.zip \
                       --suite baseline --experiment-name final_vs_best
    python evaluate.py --suite baseline --capture-steps

When the live GP module exists:
    python evaluate.py --suite gp_compatibility \
                       --gp-adapter "live:my_gp_pkg.live_adapter:LiveAdapter"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from environment import CleaningEnv

from eval.eval_config import SUITES, EvalConfig
from eval.gp_adapter import build_adapter
from eval.policies import build_policy
from eval.report import write_report
from eval.runner import run_suite
from eval.visualize import (
    plot_action_histogram,
    plot_battery_trace,
    plot_clean_ratio_distribution,
    plot_gp_calibration,
    plot_gp_residual_heatmap,
    plot_return_distribution,
    plot_termination_breakdown,
    plot_trajectory,
    plot_visit_heatmap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an eval suite on a trained model.")
    p.add_argument("--suite", type=str, default="baseline", choices=list(SUITES.keys()))
    p.add_argument("--model", type=str, default=None,
                   help="Path to .zip; defaults to <run_root>/best_model.zip")
    p.add_argument("--run-root", type=str, default="runs/ppo_v1")
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Override the suite's default name.")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--capture-steps", action="store_true")
    p.add_argument("--gp-adapter", type=str, default=None,
                   help='Override suite default; e.g. "stub" or '
                        '"live:eval.gp_adapter:ConstantLiveAdapter"')
    p.add_argument("--gp-module-arg", type=str, default=None)
    p.add_argument("--baselines", type=str, default=None,
                   help="Comma-separated; overrides suite defaults.")
    return p.parse_args()


def _resolve_model_path(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    return str(Path(args.run_root) / "best_model.zip")


def main() -> int:
    args = parse_args()
    suite_factory = SUITES[args.suite]
    base_cfg: EvalConfig = suite_factory(_resolve_model_path(args), args.run_root)

    overrides = dict(
        experiment_name=args.experiment_name,
        n_episodes=args.n_episodes,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
        capture_steps=True if args.capture_steps else None,
        gp_adapter=args.gp_adapter,
        gp_module_arg=args.gp_module_arg,
    )
    if args.baselines is not None:
        overrides["baselines"] = tuple(b.strip() for b in args.baselines.split(",") if b.strip())
    cfg = base_cfg.with_overrides(**overrides)

    output_dir = Path(cfg.run_root) / "eval" / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    cfg.to_json(output_dir / "eval_config.json")
    print(f"[evaluate] writing to {output_dir.resolve()}")

    # Build adapter + policies.
    adapter = build_adapter(cfg.gp_adapter, cfg.gp_module_arg)
    print(f"[evaluate] gp adapter: {adapter.name}")

    policies = [build_policy(cfg.model_path, cfg.deterministic)]
    target_policy = policies[0].name
    for spec in cfg.baselines:
        policies.append(build_policy(spec, cfg.deterministic))
    print(f"[evaluate] policies: {[p.name for p in policies]}")

    # Run.
    episodes = run_suite(cfg, policies, adapter)
    print(f"[evaluate] collected {len(episodes)} episodes")

    # Report.
    summary = write_report(cfg, episodes, output_dir, target_policy)

    # Plots.
    plots = output_dir / "plots"
    plot_return_distribution(episodes, plots / "return_distribution.png")
    plot_clean_ratio_distribution(episodes, plots / "clean_ratio_distribution.png")
    plot_termination_breakdown(episodes, plots / "termination_breakdown.png")

    # Per-policy plots.
    by_policy: dict[str, list] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)
    obs_map = CleaningEnv().obs_map
    for policy, eps in by_policy.items():
        safe_p = policy.replace("/", "_").replace(":", "_")
        plot_visit_heatmap(eps, obs_map, plots / f"visit_heatmap_{safe_p}.png", policy)
        plot_action_histogram(eps, plots / f"action_histogram_{safe_p}.png", policy)
        for i, ep in enumerate(eps[:cfg.representative_episodes]):
            plot_battery_trace(ep, plots / f"battery_trace_{safe_p}_ep{i}.png")
            plot_trajectory(ep, obs_map, plots / f"trajectory_{safe_p}_ep{i}.png")

    # GP-specific plots — auto-skipped when adapter is stub (§9.6 compatibility contract).
    is_live_gp = cfg.gp_adapter != "stub"
    if is_live_gp:
        for policy, eps in by_policy.items():
            safe_p = policy.replace("/", "_").replace(":", "_")
            plot_gp_calibration(eps, plots / f"gp_calibration_{safe_p}.png", policy)
        target_eps = by_policy.get(target_policy, [])
        if target_eps:
            plot_gp_residual_heatmap(
                target_eps, obs_map, plots / "gp_residual_heatmap.png"
            )
        print(f"[evaluate] GP plots written ({len(by_policy)} policies)")
    else:
        print("[evaluate] GP plots skipped (stub adapter)")

    # per_step.parquet — written only when --capture-steps was requested.
    if cfg.capture_steps:
        try:
            import pandas as pd
            rows = []
            for ep in episodes:
                for rec in ep.step_records:
                    rows.append({
                        "seed": ep.seed,
                        "policy": ep.policy,
                        "spawn_mode": ep.spawn_mode,
                        "region": ep.region or "",
                        "step": rec.step,
                        "t_global": rec.t_global,
                        "pos_r": rec.pos_r,
                        "pos_c": rec.pos_c,
                        "action": rec.action,
                        "is_collision": int(rec.is_collision),
                        "dirt_collected": rec.dirt_collected,
                        "battery": rec.battery,
                        "on_charger": int(rec.on_charger),
                        "R_total": rec.R_total,
                        "R_intrinsic": rec.R_intrinsic,
                        "R_extrinsic": rec.R_extrinsic,
                        "mu_gp_local": rec.mu_gp_local,
                        "sigma_gp_local": rec.sigma_gp_local,
                        "dirt_remaining_total": rec.dirt_remaining_total,
                    })
            if rows:
                pd.DataFrame(rows).to_parquet(
                    output_dir / "per_step.parquet", index=False
                )
                print(f"[evaluate] per_step.parquet: {len(rows):,} rows written")
            else:
                print("[evaluate] per_step.parquet: no step records collected")
        except ImportError:
            print(
                "[evaluate] WARNING: pandas not available; per_step.parquet skipped. "
                "Install pandas + pyarrow to enable step-level export."
            )

    # CI exit code: 0 if trained policy beats random by p < 0.005 on extrinsic return.
    target_returns = np.array([e.ep_return_extrinsic for e in episodes if e.policy == target_policy])
    random_returns = np.array([e.ep_return_extrinsic for e in episodes if e.policy == "random"])
    if len(target_returns) > 1 and len(random_returns) > 1:
        from eval.stats import compare_distributions
        cmp = compare_distributions(
            target_returns, random_returns,
            metric="ep_return_extrinsic",
            policy_a=target_policy, policy_b="random",
        )
        beats = (cmp.welch_p < 0.005) and (cmp.mean_a > cmp.mean_b)
        print(f"[evaluate] target vs random: Δmean={cmp.mean_diff:+.2f}, "
              f"p={cmp.welch_p:.4g} → {'PASS' if beats else 'FAIL'}")
        return 0 if beats else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
