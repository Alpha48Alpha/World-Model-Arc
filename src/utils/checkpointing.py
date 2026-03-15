"""Checkpoint saving and loading for agents and training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


class Checkpointer:
    """Saves and loads agent + training state to disk.

    Checkpoints are stored as ``.pt`` files under ``checkpoint_dir``.
    The most recent checkpoint is always saved as ``latest.pt`` and
    periodic snapshots are saved as ``step_{n}.pt``.

    Parameters
    ----------
    checkpoint_dir:
        Directory where checkpoints are written.
    save_every:
        Save a named snapshot every N calls to ``maybe_save()``.
    keep_last:
        Keep only the most recent *keep_last* named snapshots.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every: int = 100,
        keep_last: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self._call_count = 0
        self._saved_steps: list[int] = []

    def save(self, state: dict[str, Any], step: int, tag: str = "latest") -> Path:
        """Unconditionally save ``state`` to ``{tag}.pt``."""
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save(state, path)
        return path

    def maybe_save(self, state: dict[str, Any], step: int) -> Optional[Path]:
        """Save a named snapshot every ``save_every`` calls.

        Always overwrites ``latest.pt``.  Returns the snapshot path if
        a named checkpoint was written, else ``None``.
        """
        self.save(state, step, tag="latest")
        self._call_count += 1
        snapshot_path: Optional[Path] = None
        if self._call_count % self.save_every == 0:
            tag = f"step_{step}"
            snapshot_path = self.save(state, step, tag=tag)
            self._saved_steps.append(step)
            self._evict_old_snapshots()
        return snapshot_path

    def load(self, tag: str = "latest") -> dict[str, Any]:
        """Load and return a checkpoint by tag name."""
        path = self.checkpoint_dir / f"{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def latest_exists(self) -> bool:
        """Return True if a ``latest.pt`` checkpoint exists."""
        return (self.checkpoint_dir / "latest.pt").exists()

    def _evict_old_snapshots(self) -> None:
        """Delete oldest named snapshots beyond ``keep_last``."""
        while len(self._saved_steps) > self.keep_last:
            old_step = self._saved_steps.pop(0)
            old_path = self.checkpoint_dir / f"step_{old_step}.pt"
            if old_path.exists():
                old_path.unlink()
