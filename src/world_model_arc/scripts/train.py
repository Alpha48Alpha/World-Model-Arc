"""Training script for World-Model-Arc.

Usage::

    python -m world_model_arc.scripts.train --config configs/dqn_gridworld.yaml
    python -m world_model_arc.scripts.train --config configs/pg_gridworld.yaml
    python -m world_model_arc.scripts.train --config configs/dqn_continuous.yaml

All settings are driven by the YAML config; CLI flags can override them.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


def _build_env(cfg):
    """Instantiate an environment from config."""
    from world_model_arc.envs import ContinuousWorld, GridWorld

    name = cfg.env.name.lower()
    if name == "gridworld":
        goals = [tuple(g) for g in cfg.env.goals] if cfg.env.goals else None
        hazards = [tuple(h) for h in cfg.env.hazards] if cfg.env.hazards else None
        walls = [tuple(w) for w in cfg.env.walls] if cfg.env.walls else None
        return GridWorld(
            height=cfg.env.height,
            width=cfg.env.width,
            goals=goals,
            hazards=hazards,
            walls=walls,
            max_steps=cfg.env.max_steps,
            slip_prob=cfg.env.slip_prob,
            obs_mode=cfg.env.obs_mode,
        )
    elif name == "continuousworld":
        goals = [tuple(g) for g in cfg.env.goals] if cfg.env.goals else None
        hazards = [tuple(h) for h in cfg.env.hazards] if cfg.env.hazards else None
        return ContinuousWorld(
            goals=goals,
            hazards=hazards,
            max_steps=cfg.env.max_steps,
            dense_reward=cfg.env.dense_reward,
        )
    else:
        raise ValueError(f"Unknown env: {cfg.env.name}")


def _build_agent(cfg, env):
    """Instantiate an agent from config."""
    from world_model_arc.agents import DQNAgent, PolicyGradientAgent

    hidden_dims = tuple(cfg.agent.hidden_dims)
    agent_type = cfg.agent.type.lower()

    if agent_type == "dqn":
        return DQNAgent(
            obs_shape=env.observation_shape,
            n_actions=env.action_size,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_end=cfg.agent.epsilon_end,
            epsilon_decay_steps=cfg.agent.epsilon_decay_steps,
            target_update_freq=cfg.agent.target_update_freq,
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            hidden_dims=hidden_dims,
            activation=cfg.agent.activation,
            dueling=cfg.agent.dueling,
            clip_grad=cfg.agent.clip_grad,
            device=cfg.training.device,
        )
    elif agent_type in {"pg", "policygradient", "reinforce"}:
        return PolicyGradientAgent(
            obs_shape=env.observation_shape,
            n_actions=env.action_size,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            entropy_coef=cfg.agent.entropy_coef,
            hidden_dims=hidden_dims,
            activation=cfg.agent.activation,
            clip_grad=cfg.agent.clip_grad,
            device=cfg.training.device,
        )
    else:
        raise ValueError(f"Unknown agent type: {cfg.agent.type}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a World-Model-Arc agent.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--n_episodes", type=int, default=None, help="Override n_episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    args = parser.parse_args(argv)

    from world_model_arc.training import Trainer
    from world_model_arc.utils import load_config

    cfg = load_config(args.config)

    # CLI overrides
    if args.n_episodes is not None:
        cfg.training.n_episodes = args.n_episodes
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.device is not None:
        cfg.training.device = args.device

    _seed_everything(cfg.training.seed)

    env = _build_env(cfg)
    agent = _build_agent(cfg, env)

    # Resume from checkpoint
    if args.resume:
        from world_model_arc.utils import CheckpointManager

        ckpt_mgr = CheckpointManager(args.resume)
        ckpt = ckpt_mgr.load_latest()
        if ckpt:
            agent.load_state_dict(ckpt["agent"])
            print(f"[train] Resumed from episode {ckpt.get('episode', '?')}")

    trainer = Trainer(
        env=env,
        agent=agent,
        log_dir=cfg.logging.log_dir,
        experiment_name=cfg.logging.experiment_name,
        eval_interval=cfg.training.eval_interval,
        eval_episodes=cfg.training.eval_episodes,
        log_interval=cfg.training.log_interval,
        checkpoint_interval=cfg.training.checkpoint_interval,
        use_tensorboard=cfg.logging.use_tensorboard,
        seed=cfg.training.seed,
    )
    trainer.train(n_episodes=cfg.training.n_episodes)


if __name__ == "__main__":
    main()
