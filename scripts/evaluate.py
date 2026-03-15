"""Evaluation entry point.

Usage::

    python scripts/evaluate.py --config configs/dqn_grid.yaml --checkpoint runs/dqn_grid/checkpoints/latest.pt
    python scripts/evaluate.py --config configs/ppo_grid.yaml --n_eval_episodes 20 --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import load_config
from src.envs.grid_world import GridWorld
from src.envs.continuous_world import ContinuousWorld
from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.ppo_agent import PPOAgent
from src.training.evaluator import Evaluator


def _build_env(config: dict, render: bool):
    env_name = config.get("env", "GridWorld")
    kwargs = config.get("env_kwargs", {}).copy()
    seed = config.get("seed", 42)
    render_mode = "ansi" if render else None
    if env_name == "GridWorld":
        env = GridWorld(**kwargs, seed=seed, render_mode=render_mode)
    elif env_name == "ContinuousWorld":
        env = ContinuousWorld(**kwargs, seed=seed, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env: {env_name}")
    return env


def _build_agent(config: dict, env):
    agent_type = config.get("agent_type", "DQN").upper()
    kwargs = config.get("agent_kwargs", {})
    device = config.get("device", "cpu")
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        action_dim = int(env.action_space.n)
    else:
        action_dim = int(env.action_space.shape[0])

    if agent_type == "DQN":
        return DQNAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)
    elif agent_type in ("REINFORCE",):
        return REINFORCEAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)
    elif agent_type == "PPO":
        return PPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World-Model-Arc evaluation script")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint (default: latest in run_dir)",
    )
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.device:
        overrides["device"] = args.device

    config = load_config(args.config, overrides=overrides)
    env = _build_env(config, render=args.render)
    agent = _build_agent(config, env)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        run_dir = Path(config.get("run_dir", "runs/default"))
        ckpt_path = run_dir / "checkpoints" / "latest.pt"

    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        agent.load_state_dict(state["agent"])
        print(f"[evaluate] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[evaluate] No checkpoint found at {ckpt_path}; using random weights.")

    evaluator = Evaluator(env, agent)
    results = evaluator.evaluate(
        n_episodes=args.n_eval_episodes,
        render=args.render,
    )

    print("\n=== Evaluation Results ===")
    print(f"  Episodes       : {args.n_eval_episodes}")
    print(f"  Mean Return    : {results['mean_return']:.3f} ± {results['std_return']:.3f}")
    print(f"  Min / Max      : {results['min_return']:.3f} / {results['max_return']:.3f}")
    print(f"  Mean Length    : {results['mean_length']:.1f}")
    print("==========================\n")


if __name__ == "__main__":
    main()
