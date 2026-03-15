"""Evaluation / visualisation script for World-Model-Arc.

Usage::

    python -m world_model_arc.scripts.evaluate \\
        --config configs/dqn_gridworld.yaml \\
        --checkpoint runs/dqn_gridworld/checkpoints/best.pt \\
        --n_episodes 20 \\
        --render

Loads a trained agent from a checkpoint, evaluates it, prints stats,
generates a learning-curve plot, and optionally saves a GIF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained World-Model-Arc agent.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint file or a checkpoint directory.",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=20, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save a GIF of the first evaluation episode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write evaluation results (JSON + plots).",
    )
    args = parser.parse_args(argv)

    import torch

    from world_model_arc.scripts.train import _build_agent, _build_env
    from world_model_arc.training import Evaluator
    from world_model_arc.utils import CheckpointManager, load_config
    from world_model_arc.visualization import (
        plot_eval_summary,
        render_episode_gif,
        save_figure,
    )

    cfg = load_config(args.config)
    env = _build_env(cfg)
    agent = _build_agent(cfg, env)

    # Load weights
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        mgr = CheckpointManager(ckpt_path)
        state = mgr.load_best() or mgr.load_latest()
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if state is None:
        raise RuntimeError(f"No checkpoint found at {args.checkpoint}")
    agent.load_state_dict(state["agent"])
    print(f"[evaluate] Loaded checkpoint (episode {state.get('episode', '?')})")

    evaluator = Evaluator(env, agent)
    stats = evaluator.evaluate(n_episodes=args.n_episodes)

    print("\n===== Evaluation Results =====")
    for k, v in stats.items():
        if k != "episode_rewards":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.logging.log_dir) / cfg.logging.experiment_name / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stats JSON
    stats_path = output_dir / "eval_stats.json"
    json_stats = {k: v for k, v in stats.items() if k != "episode_rewards"}
    json_stats["episode_rewards"] = stats["episode_rewards"]
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(json_stats, fh, indent=2)
    print(f"\n[evaluate] Stats saved to {stats_path}")

    # Plot
    fig = plot_eval_summary(stats["episode_rewards"], title="Evaluation Rewards")
    plot_path = output_dir / "eval_rewards.png"
    save_figure(fig, plot_path)
    print(f"[evaluate] Plot saved to {plot_path}")

    # Optional GIF
    if args.render:
        traj = evaluator.run_episode(seed=99999)
        if traj["frames"]:
            gif_path = output_dir / "episode.gif"
            render_episode_gif(traj["frames"], gif_path, fps=10)
            print(f"[evaluate] GIF saved to {gif_path}")
        else:
            print("[evaluate] Environment render() not supported; skipping GIF.")


if __name__ == "__main__":
    main()
