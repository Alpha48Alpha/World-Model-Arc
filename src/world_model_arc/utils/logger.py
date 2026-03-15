"""Metrics logger with CSV and optional TensorBoard backends.

Usage::

    logger = MetricsLogger(log_dir="runs/my_exp")
    logger.log(episode=1, reward=12.3, loss=0.045, epsilon=0.9)
    logger.close()
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Optional


class MetricsLogger:
    """Logs scalar metrics to CSV and optionally to TensorBoard.

    Args:
        log_dir: Directory where logs are written.
        experiment_name: Sub-directory name for this run.
        use_tensorboard: Whether to write TensorBoard ``SummaryWriter`` events.
            Falls back gracefully if TensorBoard is not installed.
    """

    def __init__(
        self,
        log_dir: str | Path = "runs",
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
    ) -> None:
        self.run_dir = Path(log_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list[str]] = None
        self._start_time = time.time()

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))
            except Exception:
                pass  # TensorBoard not available; CSV only

    # ------------------------------------------------------------------

    def log(self, step: int, **metrics: Any) -> None:
        """Record a set of scalar metrics at *step*.

        Args:
            step: Global step counter (e.g. episode number).
            **metrics: Arbitrary keyword scalar metrics.
        """
        row = {"step": step, "wall_time": time.time() - self._start_time, **metrics}

        # Initialize CSV writer on first call
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

        # Add any new fields discovered mid-run
        new_fields = [k for k in row if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)
            # Re-open with new fieldnames (append extracols=restval)
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self._tb_writer.add_scalar(k, float(v), global_step=step)
                except (TypeError, ValueError):
                    pass

    def close(self) -> None:
        """Flush and close all writers."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    # Context-manager support
    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
