"""Experiment configuration loading and validation.

Configs are plain Python dicts loaded from YAML files.  The
``load_config`` function merges a YAML file with runtime overrides and
returns a flat dict.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Default values for all config keys
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    # Environment
    "env": "GridWorld",
    "env_kwargs": {},
    # Agent
    "agent_type": "DQN",
    "agent_kwargs": {},
    # Training
    "n_episodes": 500,
    "max_steps_per_episode": 200,
    "eval_every": 50,
    "n_eval_episodes": 5,
    # Checkpointing & logging
    "run_dir": "runs/default",
    "save_every": 50,
    "use_tensorboard": False,
    "print_every": 10,
    # Device
    "device": "cpu",
    # Reproducibility
    "seed": 42,
}


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML experiment config, merge with defaults and overrides.

    Parameters
    ----------
    path:
        Path to the YAML config file.
    overrides:
        Optional dict of values that override the YAML file's content.

    Returns
    -------
    Fully merged config dict.
    """
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        yaml_cfg = yaml.safe_load(f) or {}
    _deep_update(cfg, yaml_cfg)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    """Recursively update *base* with *updates* in-place."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
