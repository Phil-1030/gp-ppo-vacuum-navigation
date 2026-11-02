"""Post-hoc visualisation for a PPO training run.

Reads only ``runs/<run_name>/monitor.csv`` and ``metrics.csv`` and emits four
PNGs into ``runs/<run_name>/plots/``:

    episode_return.png        - per-episode return (raw + rolling mean)
    episode_length.png        - per-episode length
    clean_ratio.png           - terminal clean_ratio per episode
    loss_curves.png           - PPO policy/value/entropy losses
    loss_curves_diagnostics.png - approx_kl + clip_fraction

Usage:
    python plot_training.py runs/<run_name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Constants from environment / blueprint that show up as reference lines.
R_SUCCESS = 50.0
R_DEAD = -100.0
CURRICULUM_PHASE_STEPS = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_monitor(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "monitor.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing monitor.csv at {path}")
    # SB3 Monitor writes a JSON header line preceded by '#'.
    return pd.read_csv(path, comment="#")


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.csv at {path}")
    return pd.read_csv(path)


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _curriculum_marker(ax, n_envs: int) -> None:
    # Latest moment by which all workers have finished the curriculum phase
    # (each worker needs CURRICULUM_PHASE_STEPS env-steps individually, so
    # ``n_envs × CURRICULUM_PHASE_STEPS`` global-steps in the worst case).
    if n_envs <= 0:
        return
    boundary = n_envs * CURRICULUM_PHASE_STEPS
    ax.axvline(boundary, color="grey", linestyle=":", alpha=0.6,
               label=f"curriculum end (≤ {boundary:,} steps)")


def _plot_episode_return(df: pd.DataFrame, out: Path, window: int, n_envs: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cum_t = df["l"].cumsum()  # cumulative env-steps
    ax.scatter(cum_t, df["r"], alpha=0.3, s=8, label="episode return")
    ax.plot(cum_t, _rolling_mean(df["r"], window), color="tab:blue",
            label=f"rolling mean (w={window})")
    ax.axhline(R_SUCCESS, color="tab:green", linestyle="--", alpha=0.5,
               label=f"R_SUCCESS={R_SUCCESS:+.0f}")
    ax.axhline(R_DEAD, color="tab:red", linestyle="--", alpha=0.5,
               label=f"R_DEAD={R_DEAD:+.0f}")
    _curriculum_marker(ax, n_envs)
    ax.set_xlabel("env timesteps")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_episode_length(df: pd.DataFrame, out: Path, window: int, n_envs: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cum_t = df["l"].cumsum()
    ax.scatter(cum_t, df["l"], alpha=0.3, s=8, label="episode length")
    ax.plot(cum_t, _rolling_mean(df["l"], window), color="tab:orange",
            label=f"rolling mean (w={window})")
    _curriculum_marker(ax, n_envs)
    ax.set_xlabel("env timesteps")
    ax.set_ylabel("steps per episode")
    ax.set_title("Episode length")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_clean_ratio(df: pd.DataFrame, out: Path, window: int, n_envs: int) -> None:
    if "clean_ratio" not in df.columns:
        print("[plot_training] monitor.csv has no clean_ratio column — skipping plot.",
              file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cum_t = df["l"].cumsum()
    ax.scatter(cum_t, df["clean_ratio"], alpha=0.3, s=8, label="terminal clean_ratio")
    ax.plot(cum_t, _rolling_mean(df["clean_ratio"], window), color="tab:purple",
            label=f"rolling mean (w={window})")
    ax.axhline(1.0, color="tab:green", linestyle="--", alpha=0.5,
               label="fully clean (1.0)")
    _curriculum_marker(ax, n_envs)
    ax.set_xlabel("env timesteps")
    ax.set_ylabel("clean ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Map clean ratio at episode end")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_loss_curves(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("train/policy_gradient_loss", "policy gradient loss", "tab:blue"),
        ("train/value_loss",            "value loss",            "tab:orange"),
        ("train/entropy_loss",          "entropy loss",          "tab:green"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 9), sharex=True)
    for ax, (col, label, color) in zip(axes, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"missing column: {col}", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        ax.plot(df["timesteps"], df[col], color=color, label=label)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("env timesteps")
    fig.suptitle("PPO loss components")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_diagnostics(df: pd.DataFrame, out: Path) -> None:
    panels = [
        ("train/approx_kl",     "approx KL",     "tab:red"),
        ("train/clip_fraction", "clip fraction", "tab:brown"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 6), sharex=True)
    for ax, (col, label, color) in zip(axes, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"missing column: {col}", ha="center",
                    va="center", transform=ax.transAxes)
            continue
        ax.plot(df["timesteps"], df[col], color=color, label=label)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("env timesteps")
    fig.suptitle("PPO update diagnostics")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a PPO training run.")
    p.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    p.add_argument("--window", type=int, default=None,
                   help="Override rolling-window size (default: from config)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    cfg = _load_config(run_dir)
    window = args.window or int(cfg.get("plot_window", 50))
    n_envs = int(cfg.get("n_envs", 1))

    monitor_df = _load_monitor(run_dir)
    metrics_df = _load_metrics(run_dir)

    if len(monitor_df) == 0:
        print("[plot_training] monitor.csv has no rows yet — skipping episode plots.",
              file=sys.stderr)
    else:
        _plot_episode_return(monitor_df, plots_dir / "episode_return.png", window, n_envs)
        _plot_episode_length(monitor_df, plots_dir / "episode_length.png", window, n_envs)
        _plot_clean_ratio(monitor_df, plots_dir / "clean_ratio.png", window, n_envs)

    if len(metrics_df) == 0:
        print("[plot_training] metrics.csv has no rows yet — skipping loss plots.",
              file=sys.stderr)
    else:
        # Drop the iter-0 row if it predates the first PPO update (all NaN losses).
        loss_df = metrics_df.dropna(subset=["train/policy_gradient_loss"], how="any")
        if len(loss_df) > 0:
            _plot_loss_curves(loss_df, plots_dir / "loss_curves.png")
            _plot_diagnostics(loss_df, plots_dir / "loss_curves_diagnostics.png")

    print(f"[plot_training] wrote plots to {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
