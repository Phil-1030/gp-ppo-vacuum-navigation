"""Custom SB3 callbacks for the CleaningEnv PPO pipeline.

``MetricsCallback`` snapshots PPO's logger after every ``train()`` call and
appends one row to ``metrics.csv``. This gives ``plot_training.py`` a
deterministic CSV source for loss curves without depending on TensorBoard.

Implementation note: SB3's ``EvalCallback`` calls ``logger.dump()`` mid-rollout
when evaluation fires, which clears ``logger.name_to_value``. To avoid races we
capture losses at ``_on_rollout_start`` (which fires *after* the previous
iteration's ``train()`` and *before* the next rollout's ``EvalCallback`` can
dump), plus ``_on_training_end`` to record the very last update.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    _LOSS_FIELDS = (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
    )

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._fieldnames = ("timesteps", "iterations", *self._LOSS_FIELDS)
        self._last_logged_n_updates = -1

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # Fires after the previous iteration's ``train()`` (or before the very
        # first rollout, when there's nothing to log yet).
        self._capture()

    def _on_training_end(self) -> None:
        # Capture the final ``train()`` call's losses, which would otherwise be
        # missed because no subsequent ``_on_rollout_start`` fires.
        self._capture()

    # ── helpers ──────────────────────────────────────────────────────────

    def _capture(self) -> None:
        n_updates = int(getattr(self.model, "_n_updates", 0))
        if n_updates == 0:
            return  # no train() has run yet
        if n_updates == self._last_logged_n_updates:
            return  # already captured this update
        name_to_value = getattr(self.logger, "name_to_value", {})
        loss_row = {k: float(name_to_value.get(k, float("nan"))) for k in self._LOSS_FIELDS}
        # If every loss field is NaN, the logger was already dumped — skip.
        if all(math.isnan(v) for v in loss_row.values()):
            return
        loss_row["timesteps"] = int(self.num_timesteps)
        loss_row["iterations"] = n_updates
        self._append(loss_row)
        self._last_logged_n_updates = n_updates

    def _append(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
