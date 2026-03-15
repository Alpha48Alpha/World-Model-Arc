"""Training entry point.

Usage::

    python scripts/train.py --config configs/dqn_grid.yaml
    python scripts/train.py --config configs/ppo_grid.yaml --device cuda
    python scripts/train.py --config configs/reinforce_grid.yaml --n_episodes 1000

Overrides can be passed as ``--key value`` pairs and will take precedence
over the YAML config.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.config import load_config
from src.envs.grid_world import GridWorld
from src.envs.continuous_world import ContinuousWorld
from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import REINFORCEAgent
from src.agents.ppo_agent import PPOAgent
from src.training.trainer import Trainer


def _build_env(config: dict):
    env_name = config.get("env", "GridWorld")
    kwargs = config.get("env_kwargs", {})
    seed = config.get("seed", 42)
    if env_name == "GridWorld":
        env = GridWorld(**kwargs, seed=seed)
    elif env_name == "ContinuousWorld":
        env = ContinuousWorld(**kwargs, seed=seed)
    else:
        raise ValueError(f"Unknown env: {env_name}")
    return env


def _build_agent(config: dict, env):
    agent_type = config.get("agent_type", "DQN").upper()
    kwargs = config.get("agent_kwargs", {})
    device = config.get("device", "cpu")

    obs_dim = env.observation_space.shape[0]
    # For discrete envs the action_dim == n; for continuous we use n=2 bins
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World-Model-Arc training script")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    # Allow arbitrary key=value overrides
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ("config", "resume")
    }
    config = load_config(args.config, overrides=overrides)
    _set_seed(int(config.get("seed", 42)))

    env = _build_env(config)
    agent = _build_agent(config, env)
    trainer = Trainer(env, agent, config)

    if args.resume:
        trainer.resume()

    print(f"[train] Starting: {config['agent_type']} on {config['env']}")
    print(f"[train] Episodes: {config['n_episodes']}  |  Run dir: {config['run_dir']}")
    results = trainer.train()
    ep_returns = results["episode_returns"]
    print(
        f"[train] Done. "
        f"Final 50-ep mean return: {np.mean(ep_returns[-50:]):.3f}"
    )


if __name__ == "__main__":
    main()
