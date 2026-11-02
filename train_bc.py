"""Behavior-cloning training entry point.

Collects demonstrations from BFSGreedyPolicy, trains LR + LinearSVC classifiers,
and serializes everything to output_dir.

Usage:
    python train_bc.py
    python train_bc.py --n-episodes 200 --output-dir runs/bc
    python train_bc.py --n-episodes 50 --output-dir runs/bc_smoke --lr-C 1.0 --svm-C 0.1

Evaluation is handled separately by evaluate.py --suite bc_baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.bc_data import collect_bc_data, load_bc_data, save_bc_data
from eval.bc_models import train_and_save


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect BC data and train LR + SVM classifiers.")
    p.add_argument("--n-episodes",   type=int,   default=200,       metavar="N",
                   help="Number of teacher episodes to collect (default: 200)")
    p.add_argument("--seed-base",    type=int,   default=9000,      metavar="S",
                   help="First seed; episodes use seeds [S, S+N) (default: 9000)")
    p.add_argument("--output-dir",   type=str,   default="runs/bc", metavar="DIR",
                   help="Directory for all outputs (default: runs/bc)")
    p.add_argument("--lr-C",         type=float, default=1.0,       metavar="C",
                   help="Regularisation strength for LogisticRegression (default: 1.0)")
    p.add_argument("--svm-C",        type=float, default=0.1,       metavar="C",
                   help="Regularisation strength for LinearSVC (default: 0.1)")
    p.add_argument("--val-fraction", type=float, default=0.20,      metavar="F",
                   help="Fraction of data held out for validation (default: 0.20)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Persist hyperparameter snapshot ───────────────────────────────────
    bc_config = {
        "n_episodes":        args.n_episodes,
        "seed_base":         args.seed_base,
        "battery_threshold": 0.30,
        "output_dir":        str(output_dir),
        "lr_C":              args.lr_C,
        "svm_C":             args.svm_C,
        "val_fraction":      args.val_fraction,
    }
    (output_dir / "bc_config.json").write_text(
        json.dumps(bc_config, indent=2, sort_keys=True)
    )
    print(f"Config saved → {output_dir / 'bc_config.json'}")

    # ── 2. Collect demonstrations ─────────────────────────────────────────────
    print(f"\nCollecting demonstrations: {args.n_episodes} episodes "
          f"(seeds {args.seed_base}–{args.seed_base + args.n_episodes - 1}) …")
    t0 = time.perf_counter()
    X, y = collect_bc_data(
        n_episodes=args.n_episodes,
        seed_base=args.seed_base,
        battery_threshold=0.30,
    )
    collect_secs = time.perf_counter() - t0

    data_path = output_dir / "bc_data.npz"
    save_bc_data(X, y, data_path)
    print(f"\nData saved → {data_path}")
    print(f"  shape: X={X.shape}  y={y.shape}  "
          f"({collect_secs:.1f}s, {X.nbytes / 1e6:.1f} MB uncompressed)")

    # ── 3. Train classifiers ──────────────────────────────────────────────────
    print("\nTraining classifiers …")
    t1 = time.perf_counter()
    report = train_and_save(
        X, y,
        output_dir=output_dir,
        val_fraction=args.val_fraction,
        lr_C=args.lr_C,
        svm_C=args.svm_C,
    )
    train_secs = time.perf_counter() - t1
    print(f"  Training done in {train_secs:.1f}s")

    # ── 4. Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Behavior-cloning training summary")
    print("=" * 58)
    print(f"  Samples   train={report['n_train']:,}   val={report['n_val']:,}")

    dist = report["action_distribution"]
    print(f"  Action distribution  " +
          "  ".join(f"{a}:{float(v):.2f}" for a, v in sorted(dist.items())))

    for model_key in ("lr", "svm"):
        m = report[model_key]
        pca = "  ".join(f"{v:.3f}" for v in m["val_per_class_accuracy"])
        print(f"\n  {model_key.upper():4s}  val_accuracy={m['val_accuracy']:.4f}")
        print(f"        per-class:  [{pca}]")

    print("=" * 58)
    print(f"\nAll outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
