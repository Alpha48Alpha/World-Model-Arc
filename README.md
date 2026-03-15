# World-Model-Arc

A **production-grade AI research lab** built with modular PyTorch — featuring rich simulated worlds, policy-gradient and DQN agents, experiment configs, checkpointing, metrics logging, evaluation, and visualization so agents learn robust goal-directed behaviour from trial and error.

The codebase is designed to be extended into **world models**, **multimodal control**, and **human-in-the-loop** research.

---

## Features

| Component | Description |
|-----------|-------------|
| **Environments** | `GridWorld` (discrete, stochastic, walls/hazards/goals) and `ContinuousWorld` (2-D point-mass navigation) |
| **Agents** | `DQNAgent` (experience replay, target network, optional dueling arch) and `PolicyGradientAgent` (REINFORCE + entropy bonus) |
| **Networks** | `MLP`, `CNN`, `DuelingMLP` — all configurable |
| **World Model stub** | Learned transition + reward predictor ready for MBRL research |
| **Training** | `Trainer` class — handles rollouts, agent updates, periodic eval and checkpointing |
| **Evaluation** | `Evaluator` — deterministic rollouts, success rate, trajectory recording |
| **Logging** | `MetricsLogger` — CSV + optional TensorBoard |
| **Checkpointing** | `CheckpointManager` — rolling history, best-model tracking |
| **Visualization** | Learning curves, eval summaries, 2-D trajectory plots, animated GIF export |
| **Configs** | YAML experiment files with clean override support |

---

## Installation

```bash
# Install with dev dependencies (pytest, black, etc.)
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, PyYAML, Matplotlib, tqdm.

---

## Quick Start

### Train a DQN agent on GridWorld

```bash
python -m world_model_arc.scripts.train --config configs/dqn_gridworld.yaml
```

### Train a Policy Gradient agent on GridWorld

```bash
python -m world_model_arc.scripts.train --config configs/pg_gridworld.yaml
```

### Train a Dueling DQN on the continuous navigation world

```bash
python -m world_model_arc.scripts.train --config configs/dqn_continuous.yaml
```

### Evaluate a trained checkpoint

```bash
python -m world_model_arc.scripts.evaluate \
    --config configs/dqn_gridworld.yaml \
    --checkpoint runs/dqn_gridworld/checkpoints/best.pt \
    --n_episodes 20 \
    --render
```

### Resume training

```bash
python -m world_model_arc.scripts.train \
    --config configs/dqn_gridworld.yaml \
    --resume runs/dqn_gridworld/checkpoints
```

---

## Project Structure

```
World-Model-Arc/
├── configs/                    # YAML experiment configs
│   ├── dqn_gridworld.yaml
│   ├── pg_gridworld.yaml
│   └── dqn_continuous.yaml
├── src/world_model_arc/
│   ├── envs/                   # Simulated worlds
│   │   ├── base.py             # BaseEnv + StepResult
│   │   ├── gridworld.py        # Discrete grid world
│   │   └── continuous_world.py # Continuous 2-D navigation
│   ├── agents/                 # RL agents
│   │   ├── base.py             # BaseAgent interface
│   │   ├── dqn.py              # DQN (+ Dueling DQN)
│   │   └── policy_gradient.py  # REINFORCE
│   ├── models/                 # Neural networks
│   │   ├── networks.py         # MLP, CNN, DuelingMLP
│   │   └── world_model.py      # Learned transition model
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          # Main training loop
│   │   └── evaluator.py        # Evaluation runner
│   ├── utils/                  # Utilities
│   │   ├── config.py           # YAML config loading
│   │   ├── replay_buffer.py    # Experience replay
│   │   ├── logger.py           # CSV + TensorBoard logging
│   │   └── checkpointing.py    # Checkpoint management
│   ├── visualization/          # Plots and GIFs
│   │   └── plots.py
│   └── scripts/                # CLI entry points
│       ├── train.py
│       └── evaluate.py
├── tests/                      # pytest test suite
│   ├── test_envs.py
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_training.py
├── setup.py
├── pyproject.toml
└── requirements.txt
```

---

## Configuration

All experiment settings live in YAML files under `configs/`.  Every key maps directly to a typed dataclass field:

```yaml
env:
  name: GridWorld      # GridWorld | ContinuousWorld
  height: 8
  width: 8
  goals: [[7, 7]]
  hazards: [[3, 3]]

agent:
  type: DQN            # DQN | PG
  lr: 0.001
  dueling: false

training:
  n_episodes: 500
  seed: 42
  device: cpu

logging:
  log_dir: runs
  experiment_name: my_experiment
  use_tensorboard: true
```

---

## Python API

```python
from world_model_arc.envs import GridWorld
from world_model_arc.agents import DQNAgent
from world_model_arc.training import Trainer

env = GridWorld(height=8, width=8, goals=[(7, 7)], hazards=[(3, 3)])
agent = DQNAgent(obs_shape=env.observation_shape, n_actions=env.action_size)

trainer = Trainer(env, agent, log_dir="runs", experiment_name="my_exp")
trainer.train(n_episodes=500)
```

---

## Extending the Codebase

### Add a new environment

Subclass `BaseEnv` in `src/world_model_arc/envs/` and implement `reset`, `step`, `observation_shape`, and `action_size`.

### Add a new agent

Subclass `BaseAgent` in `src/world_model_arc/agents/` and implement `select_action`, `update`, `state_dict`, and `load_state_dict`.

### World models / MBRL

`src/world_model_arc/models/world_model.py` contains a `WorldModel` class (MLP transition + reward predictor) that can be dropped into any agent for Dyna-style planning.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT

