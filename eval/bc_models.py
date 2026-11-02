"""Train and serialize LR + LinearSVC behavior-cloning classifiers.

Public API
----------
train_and_save  — split, train, evaluate, persist models, write report JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def _per_class_accuracy(cm: np.ndarray) -> list[float]:
    """Return per-class accuracy from a confusion matrix (diagonal / row sum)."""
    row_sums = cm.sum(axis=1)
    return [
        float(cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
        for i in range(len(cm))
    ]


def _eval_model(clf, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    y_pred = clf.predict(X_val)
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])
    acc = float((y_pred == y_val).mean())
    return {
        "val_accuracy": round(acc, 6),
        "val_per_class_accuracy": [round(v, 6) for v in _per_class_accuracy(cm)],
        "confusion_matrix": cm.tolist(),
    }


def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    val_fraction: float = 0.20,
    lr_C: float = 1.0,
    svm_C: float = 0.1,
) -> dict:
    """Split data, train LR + LinearSVC, serialize models, write train_report.json.

    Models saved to:
        output_dir/lr_model.pkl   — fitted LogisticRegression
        output_dir/svm_model.pkl  — fitted LinearSVC

    Returns the report dict (also written to output_dir/train_report.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        random_state=0,
        stratify=y,
    )

    n_train, n_val = len(X_train), len(X_val)
    n_total = n_train + n_val
    action_dist = {
        str(a): round(float((y == a).sum() / n_total), 6)
        for a in range(4)
    }

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("  Training LogisticRegression …")
    lr = LogisticRegression(C=lr_C, max_iter=1000, solver="lbfgs")
    lr.fit(X_train, y_train)
    lr_metrics = _eval_model(lr, X_val, y_val)
    joblib.dump(lr, output_dir / "lr_model.pkl")
    print(f"    val_accuracy = {lr_metrics['val_accuracy']:.4f}")

    # ── LinearSVC ─────────────────────────────────────────────────────────────
    print("  Training LinearSVC …")
    svm = LinearSVC(C=svm_C, max_iter=2000, dual="auto")
    svm.fit(X_train, y_train)
    svm_metrics = _eval_model(svm, X_val, y_val)
    joblib.dump(svm, output_dir / "svm_model.pkl")
    print(f"    val_accuracy = {svm_metrics['val_accuracy']:.4f}")

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "n_train": n_train,
        "n_val": n_val,
        "action_distribution": action_dist,
        "lr": lr_metrics,
        "svm": svm_metrics,
    }
    (output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )

    return report
