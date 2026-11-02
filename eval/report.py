"""Aggregate JSON / Markdown report writer for an evaluation experiment."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .eval_config import EvalConfig
from .metrics import CSV_FIELDS, EpisodeMetrics, to_csv_row
from .stats import (
    PairwiseStats,
    compare_distributions,
    compare_proportions,
)


# Bonferroni-corrected significance threshold (5 metrics × up to 2 baselines).
_ALPHA_CORRECTED = 0.005


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs: list[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")


def write_episodes_csv(episodes: list[EpisodeMetrics], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for e in episodes:
            w.writerow(to_csv_row(e))


def aggregate(episodes: list[EpisodeMetrics]) -> dict:
    """Build a {policy: aggregate_dict} summary for all numeric fields."""
    by_policy: dict[str, list[EpisodeMetrics]] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)

    summary = {}
    for policy, eps in by_policy.items():
        n = len(eps)
        causes = Counter(e.terminated_cause for e in eps)

        returns_ext = [e.ep_return_extrinsic for e in eps]
        clean_ratios = [e.clean_ratio_final for e in eps]

        summary[policy] = {
            "n_episodes": n,
            "ep_length_mean":           _safe_mean([e.ep_length for e in eps]),
            "ep_length_std":            _safe_std([e.ep_length for e in eps]),
            "ep_return_extrinsic_mean": _safe_mean(returns_ext),
            "ep_return_extrinsic_std":  _safe_std(returns_ext),
            "ep_return_extrinsic_median": float(np.median(returns_ext)) if returns_ext else float("nan"),
            "ep_return_extrinsic_iqr":  float(np.percentile(returns_ext, 75) - np.percentile(returns_ext, 25)) if returns_ext else float("nan"),
            "ep_return_total_mean":     _safe_mean([e.ep_return_total for e in eps]),
            "clean_ratio_mean":         _safe_mean(clean_ratios),
            "clean_ratio_std":          _safe_std(clean_ratios),
            "clean_ratio_median":       float(np.median(clean_ratios)) if clean_ratios else float("nan"),
            "clean_ratio_iqr":          float(np.percentile(clean_ratios, 75) - np.percentile(clean_ratios, 25)) if clean_ratios else float("nan"),
            "cells_cleaned_mean":       _safe_mean([e.cells_cleaned_unique for e in eps]),
            "n_collisions_mean":        _safe_mean([e.n_collisions for e in eps]),
            "n_charger_visits_mean":    _safe_mean([e.n_charger_visits for e in eps]),
            "total_charge_steps_mean":  _safe_mean([e.total_charge_steps for e in eps]),
            "min_battery_mean":         _safe_mean([e.min_battery for e in eps if not math.isinf(e.min_battery)]),
            "time_below_30pct_mean":    _safe_mean([e.time_below_30pct for e in eps]),
            "first_charger_visit_step_mean":
                _safe_mean([e.first_charger_visit_step for e in eps
                            if not math.isnan(e.first_charger_visit_step)]),
            "success_rate":     causes.get("SUCCESS", 0) / n,
            "death_rate":       causes.get("DEAD", 0) / n,
            "truncation_rate":  causes.get("TRUNCATED", 0) / n,
            "charger_visit_rate": sum(1 for e in eps if e.n_charger_visits > 0) / n,
            "termination_counts": dict(causes),
            "maintenance_ratio_mean": _safe_mean([e.maintenance_ratio for e in eps]),
            "maintenance_ratio_std":  _safe_std([e.maintenance_ratio for e in eps]),
            "peak_clean_ratio_mean":  _safe_mean([e.peak_clean_ratio for e in eps]),
            "coverage_rate_mean":     _safe_mean([e.coverage_rate for e in eps]),
        }
    return summary


def write_summary_json(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def _group_by_policy_region(
    episodes: list[EpisodeMetrics],
) -> dict[tuple[str, str], list[EpisodeMetrics]]:
    """Group episodes by (policy, region) for room_breakdown analysis."""
    out: dict[tuple[str, str], list[EpisodeMetrics]] = {}
    for e in episodes:
        if e.region:
            out.setdefault((e.policy, e.region), []).append(e)
    return out


def pairwise_tests(episodes: list[EpisodeMetrics], target: str) -> list[PairwiseStats]:
    """Compare every other policy against ``target``."""
    by_policy: dict[str, list[EpisodeMetrics]] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)

    if target not in by_policy:
        return []

    target_eps = by_policy[target]
    out: list[PairwiseStats] = []

    target_returns = np.array([e.ep_return_extrinsic for e in target_eps])
    target_clean = np.array([e.clean_ratio_final for e in target_eps])
    target_chg = np.array([e.n_charger_visits for e in target_eps])
    target_succ = sum(1 for e in target_eps if e.terminated_cause == "SUCCESS")

    for other, eps in by_policy.items():
        if other == target:
            continue
        other_returns = np.array([e.ep_return_extrinsic for e in eps])
        other_clean = np.array([e.clean_ratio_final for e in eps])
        other_chg = np.array([e.n_charger_visits for e in eps])
        other_succ = sum(1 for e in eps if e.terminated_cause == "SUCCESS")
        out.append(compare_distributions(
            target_returns, other_returns,
            metric="ep_return_extrinsic", policy_a=target, policy_b=other,
        ))
        out.append(compare_distributions(
            target_clean, other_clean,
            metric="clean_ratio_final", policy_a=target, policy_b=other,
        ))
        out.append(compare_distributions(
            target_chg, other_chg,
            metric="n_charger_visits", policy_a=target, policy_b=other,
        ))
        out.append(compare_proportions(
            target_succ, len(target_eps), other_succ, len(eps),
            metric="success_rate", policy_a=target, policy_b=other,
        ))
    return out


def render_markdown(
    cfg: EvalConfig,
    summary: dict,
    pairwise: list[PairwiseStats],
    target_policy: str,
    *,
    episodes: list[EpisodeMetrics] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation report — {cfg.experiment_name}")
    lines.append("")
    lines.append(f"- Model: `{cfg.model_path}`")
    lines.append(f"- Run root: `{cfg.run_root}`")
    lines.append(f"- Episodes: {cfg.n_episodes}")
    lines.append(f"- Spawn mode: `{cfg.spawn_mode}`")
    lines.append(f"- GP adapter: `{cfg.gp_adapter}`")
    lines.append("")

    # Plain-language summary for the trained policy.
    if target_policy in summary:
        s = summary[target_policy]
        n = s["n_episodes"]
        clean_mean = s["clean_ratio_mean"]
        success = int(round(s["success_rate"] * n))
        death = int(round(s["death_rate"] * n))
        chg_rate = s["charger_visit_rate"]
        lines.append("## Plain-language summary")
        lines.append("")
        lines.append(
            f"Trained policy `{target_policy}` achieves a mean clean ratio of "
            f"**{clean_mean:.3f}** (mean ep_length "
            f"{s['ep_length_mean']:.1f}). "
            f"**{success} / {n}** episodes ended in SUCCESS and "
            f"**{death} / {n}** ended in DEAD. The charger was visited in "
            f"**{chg_rate:.0%}** of episodes "
            f"(mean visits/episode: {s['n_charger_visits_mean']:.2f})."
        )
        lines.append("")

    # Aggregate table.
    lines.append("## Aggregate metrics per policy")
    lines.append("")
    cols = [
        "n_episodes", "ep_length_mean", "ep_return_extrinsic_mean",
        "clean_ratio_mean", "maintenance_ratio_mean", "peak_clean_ratio_mean",
        "coverage_rate_mean", "n_charger_visits_mean",
        "success_rate", "death_rate", "truncation_rate", "charger_visit_rate",
    ]
    header = "| policy | " + " | ".join(cols) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for p, s in summary.items():
        row = [p] + [f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c]) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-region breakdown (room_breakdown suite only).
    if episodes:
        region_groups = _group_by_policy_region(episodes)
        if region_groups:
            lines.append("## Per-region breakdown (room_breakdown suite)")
            lines.append("")
            lines.append("| policy | region | n_ep | clean_ratio_mean | clean_ratio_median | death_rate | charger_visit_rate |")
            lines.append("|--------|--------|------|------------------|--------------------|------------|-------------------|")
            for (policy, region), eps in sorted(region_groups.items()):
                n = len(eps)
                causes = Counter(e.terminated_cause for e in eps)
                cr_vals = [e.clean_ratio_final for e in eps]
                cr_mean = float(np.mean(cr_vals)) if cr_vals else float("nan")
                cr_med = float(np.median(cr_vals)) if cr_vals else float("nan")
                death_r = causes.get("DEAD", 0) / n if n else float("nan")
                chg_r = sum(1 for e in eps if e.n_charger_visits > 0) / n if n else float("nan")
                lines.append(
                    f"| {policy} | {region} | {n} | {cr_mean:.3f} | "
                    f"{cr_med:.3f} | {death_r:.3f} | {chg_r:.3f} |"
                )
            lines.append("")

    # Pairwise comparisons.
    if pairwise:
        lines.append(f"## Pairwise tests vs `{target_policy}`")
        lines.append(f"Significance threshold (Bonferroni-corrected): **p < {_ALPHA_CORRECTED}**.")
        lines.append("")
        lines.append("| metric | vs | mean(target) | mean(other) | diff | 95% CI | t/z | p (Welch/z) | p (Mann-Whitney) | sig |")
        lines.append("|--------|----|--------------|-------------|------|--------|-----|-------------|------------------|-----|")
        for r in pairwise:
            sig = "✓" if (not math.isnan(r.welch_p) and r.welch_p < _ALPHA_CORRECTED) else ""
            ci = (f"[{r.bootstrap_lo:+.3f}, {r.bootstrap_hi:+.3f}]"
                  if not math.isnan(r.bootstrap_lo) else "—")
            mw = f"{r.mw_p:.4f}" if not math.isnan(r.mw_p) else "—"
            lines.append(
                f"| {r.metric} | {r.policy_b} | {r.mean_a:.3f} | {r.mean_b:.3f} | "
                f"{r.mean_diff:+.3f} | {ci} | {r.welch_t:+.2f} | {r.welch_p:.4f} | {mw} | {sig} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_report(
    cfg: EvalConfig,
    episodes: list[EpisodeMetrics],
    output_dir: Path,
    target_policy: str,
) -> dict:
    """Write episodes.csv, summary.json, report.md. Returns the summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_episodes_csv(episodes, output_dir / "episodes.csv")
    summary = aggregate(episodes)
    write_summary_json(summary, output_dir / "summary.json")
    pw = pairwise_tests(episodes, target_policy)
    md = render_markdown(cfg, summary, pw, target_policy, episodes=episodes)
    (output_dir / "report.md").write_text(md)
    return summary
