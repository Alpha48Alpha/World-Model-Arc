"""Configuration management for World-Model-Arc experiments.

Uses YAML files as the primary config format.  All fields have sensible
defaults so that researchers can write minimal config overrides.

Example config file::

    env:
      name: GridWorld
      height: 8
      width: 8
    agent:
      type: DQN
      lr: 1e-3
    training:
      n_episodes: 1000
      batch_size: 64
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs (dataclasses for IDE support and type checking)
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    name: str = "GridWorld"
    height: int = 8
    width: int = 8
    obs_mode: str = "flat"
    max_steps: int = 200
    slip_prob: float = 0.0
    dense_reward: bool = True
    goals: list[list[int]] = field(default_factory=list)
    hazards: list[list[int]] = field(default_factory=list)
    walls: list[list[int]] = field(default_factory=list)


@dataclass
class AgentConfig:
    type: str = "DQN"
    lr: float = 1e-3
    gamma: float = 0.99
    # DQN-specific
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_freq: int = 1_000
    buffer_capacity: int = 50_000
    batch_size: int = 64
    # Network
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    dueling: bool = False
    # Policy gradient specific
    entropy_coef: float = 0.01
    clip_grad: float = 1.0


@dataclass
class TrainingConfig:
    n_episodes: int = 500
    eval_interval: int = 50
    eval_episodes: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 100
    seed: int = 42
    device: str = "cpu"


@dataclass
class LoggingConfig:
    log_dir: str = "runs"
    experiment_name: str = "experiment"
    use_tensorboard: bool = True


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Raw dict retained for extension fields not captured above
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Keys missing from the YAML will fall back to dataclass defaults.

    Args:
        path: Path to a ``.yaml`` or ``.yml`` config file.

    Returns:
        Populated :class:`ExperimentConfig`.
    """
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = ExperimentConfig()

    if "env" in raw:
        env_dict = {k: v for k, v in raw["env"].items()}
        for k, v in env_dict.items():
            if hasattr(cfg.env, k):
                setattr(cfg.env, k, v)

    if "agent" in raw:
        for k, v in raw["agent"].items():
            if hasattr(cfg.agent, k):
                setattr(cfg.agent, k, v)

    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)

    if "logging" in raw:
        for k, v in raw["logging"].items():
            if hasattr(cfg.logging, k):
                setattr(cfg.logging, k, v)

    cfg._extra = {k: v for k, v in raw.items() if k not in {"env", "agent", "training", "logging"}}

    return cfg
