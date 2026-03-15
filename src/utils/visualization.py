"""Training curve and episode visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Guard matplotlib import so the module can be imported in headless environments
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False


def _check_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation.  "
            "Install it with:  pip install matplotlib"
        )


def _smooth(values: Sequence[float], window: int) -> np.ndarray:
    """Uniform moving-average smoothing."""
    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curves(
    metrics: dict[str, list[float]],
    title: str = "Training Curves",
    smooth_window: int = 10,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot one subplot per metric key.

    Parameters
    ----------
    metrics:
        Dict mapping metric name → list of scalar values.
    title:
        Figure super-title.
    smooth_window:
        Moving-average window for smoothing (1 = no smoothing).
    save_path:
        If provided, save the figure to this path instead of displaying.
    """
    _check_mpl()

    n = len(metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for ax, (name, values) in zip(axes.flatten(), metrics.items()):
        xs = np.arange(len(values))
        ax.plot(xs, values, alpha=0.3, color="steelblue", label="raw")
        smoothed = _smooth(values, smooth_window)
        if len(smoothed) > 0:
            xs_s = np.arange(len(smoothed)) + (smooth_window - 1) // 2
            ax.plot(xs_s, smoothed, color="steelblue", linewidth=2, label="smoothed")
        ax.set_xlabel("Episode")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_episode_returns(
    returns: list[float],
    title: str = "Episode Returns",
    smooth_window: int = 20,
    save_path: Optional[str | Path] = None,
) -> None:
    """Convenience wrapper for plotting a single episode-return series."""
    plot_training_curves(
        metrics={"Episode Return": returns},
        title=title,
        smooth_window=smooth_window,
        save_path=save_path,
    )
