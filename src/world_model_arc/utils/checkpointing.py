"""Checkpoint manager: save and restore agent / training state.

Maintains a fixed-size history of the most-recent checkpoints and keeps
track of the *best* model by a monitored metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointManager:
    """Save/load PyTorch model + optimiser checkpoints.

    Args:
        checkpoint_dir: Root directory for checkpoint files.
        max_to_keep: How many recent checkpoints to retain on disk.
            Older ones are automatically deleted (except the best).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

        self._history: list[Path] = []
        self._best_path: Optional[Path] = None
        self._best_metric: float = float("-inf")
        self._meta_path = self.checkpoint_dir / "meta.json"
        self._load_meta()

    # ------------------------------------------------------------------

    def save(
        self,
        state: dict[str, Any],
        step: int,
        metric: Optional[float] = None,
    ) -> Path:
        """Persist *state* to disk.

        Args:
            state: Arbitrary dictionary (must be serialisable by
                :func:`torch.save`).
            step: Training step / episode number used to name the file.
            metric: If provided, tracks the best checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        path = self.checkpoint_dir / f"ckpt_{step:08d}.pt"
        torch.save(state, path)
        self._history.append(path)

        # Track best
        if metric is not None and metric > self._best_metric:
            self._best_metric = metric
            self._best_path = path
            best_link = self.checkpoint_dir / "best.pt"
            if best_link.exists():
                best_link.unlink()
            try:
                best_link.symlink_to(path.name)
            except OSError:
                # Symlinks may not be supported (e.g. some Windows setups)
                torch.save(state, best_link)

        # Prune old checkpoints
        while len(self._history) > self.max_to_keep:
            old = self._history.pop(0)
            if old != self._best_path and old.exists():
                old.unlink()

        self._save_meta()
        return path

    def load_latest(self) -> Optional[dict[str, Any]]:
        """Load the most recent checkpoint, or *None* if none exist."""
        if not self._history:
            return None
        return torch.load(self._history[-1], map_location="cpu", weights_only=False)

    def load_best(self) -> Optional[dict[str, Any]]:
        """Load the checkpoint with the highest tracked metric."""
        best_link = self.checkpoint_dir / "best.pt"
        if best_link.exists():
            return torch.load(best_link, map_location="cpu", weights_only=False)
        return None

    def load_path(self, path: str | Path) -> dict[str, Any]:
        """Load a specific checkpoint file."""
        return torch.load(str(path), map_location="cpu", weights_only=False)

    # ------------------------------------------------------------------

    def _save_meta(self) -> None:
        meta = {
            "history": [str(p) for p in self._history],
            "best_path": str(self._best_path) if self._best_path else None,
            "best_metric": self._best_metric,
        }
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def _load_meta(self) -> None:
        if not self._meta_path.exists():
            return
        with open(self._meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        self._history = [Path(p) for p in meta.get("history", []) if Path(p).exists()]
        bp = meta.get("best_path")
        self._best_path = Path(bp) if bp and Path(bp).exists() else None
        self._best_metric = meta.get("best_metric", float("-inf"))
