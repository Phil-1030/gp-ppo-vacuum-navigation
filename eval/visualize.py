"""Plot generation for an evaluation experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from environment import (  # noqa: E402
    BLOCKED_CELL_VALUES,
    CHARGER,
    GRID_H,
    GRID_W,
    SUCCESS_THRESHOLD,
)

from .metrics import EpisodeMetrics  # noqa: E402


_TERM_ORDER = ("SUCCESS", "TRUNCATED", "DEAD", "TERMINATED_OTHER", "INCOMPLETE")
_TERM_COLORS = {
    "SUCCESS":          "tab:green",
    "TRUNCATED":        "tab:orange",
    "DEAD":             "tab:red",
    "TERMINATED_OTHER": "tab:gray",
    "INCOMPLETE":       "tab:purple",
}


def _by_policy(episodes: list[EpisodeMetrics]) -> dict[str, list[EpisodeMetrics]]:
    out: dict[str, list[EpisodeMetrics]] = {}
    for e in episodes:
        out.setdefault(e.policy, []).append(e)
    return out


def plot_return_distribution(episodes: list[EpisodeMetrics], out: Path) -> None:
    by_policy = _by_policy(episodes)
    labels = list(by_policy.keys())
    data = [[e.ep_return_extrinsic for e in by_policy[p]] for p in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("ep_return_extrinsic")
    ax.set_title("Episode return (extrinsic) per policy")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_clean_ratio_distribution(episodes: list[EpisodeMetrics], out: Path) -> None:
    by_policy = _by_policy(episodes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 21)
    for policy, eps in by_policy.items():
        ax.hist([e.clean_ratio_final for e in eps], bins=bins, alpha=0.45,
                label=policy)
    ax.axvline(1.0 - SUCCESS_THRESHOLD, color="black", linestyle="--",
               alpha=0.6, label=f"success threshold = {1.0 - SUCCESS_THRESHOLD}")
    ax.set_xlabel("clean_ratio_final")
    ax.set_ylabel("episodes")
    ax.set_title("Final map clean ratio")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_termination_breakdown(episodes: list[EpisodeMetrics], out: Path) -> None:
    by_policy = _by_policy(episodes)
    labels = list(by_policy.keys())
    counts = {cause: [] for cause in _TERM_ORDER}
    for p in labels:
        cs = {cause: 0 for cause in _TERM_ORDER}
        for e in by_policy[p]:
            cs[e.terminated_cause] = cs.get(e.terminated_cause, 0) + 1
        for cause in _TERM_ORDER:
            counts[cause].append(cs.get(cause, 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(labels))
    for cause in _TERM_ORDER:
        vals = np.array(counts[cause], dtype=np.float64)
        if vals.sum() == 0:
            continue
        ax.bar(labels, vals, bottom=bottom, label=cause,
               color=_TERM_COLORS[cause])
        bottom += vals
    ax.set_ylabel("episodes")
    ax.set_title("Termination cause breakdown")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_visit_heatmap(
    episodes: list[EpisodeMetrics],
    obs_map: np.ndarray,
    out: Path,
    policy: str,
) -> None:
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    for e in episodes:
        grid += e.visit_counts
    if len(episodes):
        grid /= len(episodes)
    blocked = np.isin(obs_map, BLOCKED_CELL_VALUES)
    masked = np.ma.masked_where(blocked, grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(masked, cmap="viridis", origin="upper")
    ax.set_title(f"Mean cell-visit count — {policy}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mark charger.
    cr, cc = np.where(obs_map == CHARGER)
    if len(cr) > 0:
        ax.scatter(cc, cr, marker="*", s=160, color="yellow",
                   edgecolors="black", label="charger")
        ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_action_histogram(episodes: list[EpisodeMetrics], out: Path, policy: str) -> None:
    if not episodes:
        return
    counts = np.zeros_like(episodes[0].action_counts)
    for e in episodes:
        counts = counts + e.action_counts
    total = counts.sum()
    if total == 0:
        return
    fractions = counts / total
    labels = ["Up", "Down", "Left", "Right"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, fractions, color="tab:blue", alpha=0.7)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, max(0.5, fractions.max() * 1.2))
    ax.set_title(f"Action distribution — {policy}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_battery_trace(ep: EpisodeMetrics, out: Path) -> None:
    if not ep.step_records:
        return
    steps = [r.step for r in ep.step_records]
    battery = [r.battery for r in ep.step_records]
    on_charger = np.array([r.on_charger for r in ep.step_records], dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, battery, color="tab:blue", label="battery")
    ax.fill_between(steps, 0, 1, where=on_charger, alpha=0.15, color="tab:green",
                    label="on charger")
    ax.axhline(0.30, color="tab:red", linestyle="--", alpha=0.5,
               label="α_E gate (E=0.30)")
    ax.set_xlabel("step")
    ax.set_ylabel("battery")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Battery trace — {ep.policy} (seed {ep.seed}, {ep.terminated_cause})")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trajectory(ep: EpisodeMetrics, obs_map: np.ndarray, out: Path) -> None:
    if not ep.step_records:
        return
    rs = np.array([r.pos_r for r in ep.step_records])
    cs = np.array([r.pos_c for r in ep.step_records])
    blocked = np.isin(obs_map, BLOCKED_CELL_VALUES)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(blocked, cmap="Greys", alpha=0.4, origin="upper")
    ax.plot(cs, rs, color="tab:blue", linewidth=1.0, alpha=0.7)
    ax.scatter(cs[0], rs[0], color="tab:green", s=80, marker="o",
               label="start", zorder=5)
    ax.scatter(cs[-1], rs[-1], color="tab:red", s=80, marker="X",
               label=f"end ({ep.terminated_cause})", zorder=5)
    cr, cc = np.where(obs_map == CHARGER)
    if len(cr):
        ax.scatter(cc, cr, marker="*", s=160, color="yellow",
                   edgecolors="black", label="charger", zorder=4)
    ax.set_title(f"Trajectory — {ep.policy} (seed {ep.seed})")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── GP-specific plots (§9.6) — auto-skipped when gp_adapter == "stub" ────────


def plot_gp_calibration(
    episodes: list[EpisodeMetrics], out: Path, policy: str
) -> None:
    """Scatter of σ_gp (end-of-episode, per visited cell) vs visit count.

    Each point represents one (episode, cell) pair where the cell was visited
    at least once.  An information-greedy agent should visit high-σ cells more
    often, producing a positive correlation.
    """
    sigma_vals: list[float] = []
    visit_vals: list[int] = []
    for ep in episodes:
        if ep.gp_sigma_final is None:
            continue
        for r in range(GRID_H):
            for c in range(GRID_W):
                cnt = int(ep.visit_counts[r, c])
                if cnt > 0:
                    sigma_vals.append(float(ep.gp_sigma_final[r, c]))
                    visit_vals.append(cnt)
    if not sigma_vals:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sigma_vals, visit_vals, alpha=0.15, s=8, color="tab:blue",
               rasterized=True)
    ax.set_xlabel("σ_gp at cell (end of episode)")
    ax.set_ylabel("cell visit count (episode total)")
    ax.set_title(f"GP calibration — {policy}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gp_residual_heatmap(
    episodes: list[EpisodeMetrics], obs_map: np.ndarray, out: Path
) -> None:
    """Heatmap of |μ_gp − true_dirt| averaged across all episodes.

    Uses the end-of-episode GP posterior and ground-truth dirt map captured
    in ``EpisodeMetrics.gp_mu_final`` / ``dirt_map_final``.
    """
    residual_sum = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    count = 0
    for ep in episodes:
        if ep.gp_mu_final is None or ep.dirt_map_final is None:
            continue
        residual_sum += np.abs(
            ep.gp_mu_final.astype(np.float64)
            - ep.dirt_map_final.astype(np.float64)
        )
        count += 1
    if count == 0:
        return

    avg_residual = residual_sum / count
    blocked = np.isin(obs_map, BLOCKED_CELL_VALUES)
    masked = np.ma.masked_where(blocked, avg_residual)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(masked, cmap="hot", origin="upper", vmin=0)
    ax.set_title("|μ_gp − true_dirt| residual heatmap\n(mean over episodes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
