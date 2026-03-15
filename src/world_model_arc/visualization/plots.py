"""Visualization utilities for World-Model-Arc.

All functions return :class:`matplotlib.figure.Figure` objects so callers
can display or save them as needed.  We use a non-interactive backend by
default so plots work in headless CI environments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless – must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


def plot_learning_curve(
    rewards: Sequence[float],
    window: int = 20,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Total Reward",
) -> plt.Figure:
    """Plot episode rewards with a smoothed moving average.

    Args:
        rewards: Per-episode total rewards.
        window: Rolling window size for the smoothed curve.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rewards_arr = np.array(rewards, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, rewards_arr, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw")

    if len(rewards_arr) >= window:
        smoothed = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            smoothed,
            color="steelblue",
            linewidth=2,
            label=f"MA-{window}",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


def plot_eval_summary(
    eval_rewards: Sequence[float],
    eval_steps: Optional[Sequence[int]] = None,
    title: str = "Evaluation Rewards",
) -> plt.Figure:
    """Bar / line chart of evaluation checkpoint rewards.

    Args:
        eval_rewards: Mean evaluation reward at each eval checkpoint.
        eval_steps: Episode numbers at which each eval was done.
            If *None*, uses integer indices.
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    x = np.arange(len(eval_rewards)) if eval_steps is None else np.array(eval_steps)
    y = np.array(eval_rewards, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, y, width=max(1, (x[-1] - x[0]) / len(x) * 0.8) if len(x) > 1 else 0.8,
           color="mediumseagreen", alpha=0.7, label="Eval reward")
    ax.plot(x, y, "o-", color="darkgreen", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------


def plot_trajectory(
    positions: Sequence[tuple[float, float]],
    goals: Optional[Sequence[tuple[float, float]]] = None,
    hazards: Optional[Sequence[tuple[float, float]]] = None,
    title: str = "Agent Trajectory",
) -> plt.Figure:
    """Plot a 2-D agent trajectory with optional goal/hazard markers.

    Args:
        positions: Sequence of (x, y) positions visited.
        goals: Optional list of goal positions to mark.
        hazards: Optional list of hazard positions to mark.
        title: Figure title.

    Returns:
        :class:`matplotlib.figure.Figure`.
    """
    xs, ys = zip(*positions) if positions else ([], [])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "-o", markersize=3, linewidth=1, color="steelblue", alpha=0.7, label="Trajectory")
    if xs:
        ax.plot(xs[0], ys[0], "gs", markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], "r^", markersize=10, label="End")

    if goals:
        for gx, gy in goals:
            ax.add_patch(plt.Circle((gx, gy), 0.05, color="green", alpha=0.4))
        gxs, gys = zip(*goals)
        ax.scatter(gxs, gys, marker="*", s=200, color="green", zorder=5, label="Goal")

    if hazards:
        for hx, hy in hazards:
            ax.add_patch(plt.Circle((hx, hy), 0.05, color="red", alpha=0.3))
        hxs, hys = zip(*hazards)
        ax.scatter(hxs, hys, marker="X", s=150, color="red", zorder=5, label="Hazard")

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------


def render_episode_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 10,
) -> None:
    """Save a sequence of RGB frames as an animated GIF.

    Falls back gracefully (saves first frame as PNG) if *imageio* or
    *Pillow* is not installed.

    Args:
        frames: List of ``(H, W, 3)`` uint8 RGB arrays.
        path: Output file path.  Should end in ``.gif``.
        fps: Frames per second.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image  # type: ignore

        imgs = [Image.fromarray(f) for f in frames]
        if imgs:
            imgs[0].save(
                path,
                save_all=True,
                append_images=imgs[1:],
                loop=0,
                duration=max(1, 1000 // fps),
            )
    except ImportError:
        # Fallback: save first frame as PNG
        fallback = path.with_suffix(".png")
        if frames:
            fig, ax = plt.subplots()
            ax.imshow(frames[0])
            ax.axis("off")
            fig.savefig(fallback, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to *path*, creating parent dirs as needed.

    Args:
        fig: The figure to save.
        path: Output file path (e.g. ``runs/exp/learning_curve.png``).
        dpi: Image resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
