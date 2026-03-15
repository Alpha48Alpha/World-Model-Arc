"""Structured metrics logging to console, CSV, and optional TensorBoard."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Optional


class MetricsLogger:
    """Lightweight metrics recorder that writes to CSV and (optionally)
    TensorBoard.

    Usage::

        logger = MetricsLogger(log_dir="runs/dqn_grid")
        logger.log(episode=1, episode_return=-3.5, length=120, loss=0.042)
        logger.close()

    Parameters
    ----------
    log_dir:
        Directory where ``metrics.csv`` and TensorBoard events are stored.
    use_tensorboard:
        Whether to write TensorBoard summaries (requires ``tensorboard``).
    print_every:
        Print a summary line every N ``log()`` calls (0 = never).
    """

    def __init__(
        self,
        log_dir: str | Path,
        use_tensorboard: bool = False,
        print_every: int = 10,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.print_every = print_every
        self._call_count = 0
        self._start_time = time.time()

        # CSV writer
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._fieldnames: list[str] = []

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                pass

    def log(self, **metrics: Any) -> None:
        """Record a row of scalar metrics.

        Keyword arguments become column names.  The wall-clock
        ``elapsed_s`` is appended automatically.
        """
        row: dict[str, Any] = dict(metrics)
        row["elapsed_s"] = round(time.time() - self._start_time, 2)

        # Lazily initialise CSV header on first log call
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            step = int(row.get("episode", row.get("step", self._call_count)))
            for k, v in row.items():
                if isinstance(v, (int, float)) and k not in ("episode", "step"):
                    self._tb_writer.add_scalar(k, float(v), global_step=step)

        self._call_count += 1
        if self.print_every > 0 and self._call_count % self.print_every == 0:
            self._print_row(row)

    def _print_row(self, row: dict[str, Any]) -> None:
        parts = [f"{k}={v}" for k, v in row.items()]
        print(" | ".join(parts))

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def read_all(self) -> list[dict[str, Any]]:
        """Return all logged rows as a list of dicts (re-reads CSV)."""
        if not self._csv_path.exists():
            return []
        with self._csv_path.open() as f:
            reader = csv.DictReader(f)
            return list(reader)
