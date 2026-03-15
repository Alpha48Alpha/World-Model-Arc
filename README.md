# World-Model-Arc

A **production-grade AI research lab** built on PyTorch. The project provides a modular reinforcement-learning system where agents learn goal-directed behaviour through trial and error in rich simulated worlds. The codebase is designed to be easy to extend into world models, multimodal control, and human-in-the-loop research.

---

## ✨ Features

| Area | What's included |
|------|----------------|
| **Environments** | `GridWorld` (discrete, 2-D navigation) · `ContinuousWorld` (2-D point-mass navigation) · `NormalizeObservation` · `FrameStack` wrappers |
| **Agents** | `DQNAgent` (Double DQN + replay buffer) · `REINFORCEAgent` (policy gradient + optional baseline) · `PPOAgent` (clipped PPO + GAE) |
| **Models** | `MLP` · `CNN` · `ActorCritic` shared backbone |
| **Memory** | `ReplayBuffer` (uniform, circular) · `RolloutStorage` (on-policy, GAE) |
| **Training** | `Trainer` – unified loop for all agent types, eval hooks, auto-saves |
| **Evaluation** | `Evaluator` – deterministic rollouts, summary statistics |
| **Logging** | `MetricsLogger` – CSV + optional TensorBoard |
| **Checkpointing** | `Checkpointer` – latest + named snapshots, auto-eviction |
| **Visualization** | `plot_training_curves` · `plot_episode_returns` (matplotlib) |
| **Configs** | YAML experiment configs for DQN / REINFORCE / PPO on Grid and Continuous worlds |
| **Scripts** | `scripts/train.py` · `scripts/evaluate.py` |

---

## 📁 Repository Layout

```
World-Model-Arc/
├── src/
│   ├── envs/           # Simulated worlds + wrappers
│   ├── agents/         # DQN, REINFORCE, PPO
│   ├── models/         # MLP, CNN, ActorCritic, ReplayBuffer, RolloutStorage
│   ├── training/       # Trainer, Evaluator
│   ├── utils/          # MetricsLogger, Checkpointer, Visualization
│   └── config.py       # YAML config loading
├── configs/            # Experiment YAML files
├── tests/              # Pytest test suite (77 tests)
├── scripts/
│   ├── train.py        # Training entry point
│   └── evaluate.py     # Evaluation entry point
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

### Train

```bash
# DQN on GridWorld
python scripts/train.py --config configs/dqn_grid.yaml

# REINFORCE with baseline on GridWorld
python scripts/train.py --config configs/reinforce_grid.yaml

# PPO on GridWorld
python scripts/train.py --config configs/ppo_grid.yaml

# PPO on ContinuousWorld
python scripts/train.py --config configs/ppo_continuous.yaml

# Override any config key on the command line
python scripts/train.py --config configs/dqn_grid.yaml --n_episodes 1000 --device cuda

# Resume from latest checkpoint
python scripts/train.py --config configs/dqn_grid.yaml --resume
```

Training artefacts are written to the `run_dir` specified in the YAML:
- `metrics.csv`   – per-episode metrics
- `training_curves.png` – smoothed training curves
- `checkpoints/latest.pt` + `checkpoints/step_N.pt` – model snapshots

### Evaluate

```bash
python scripts/evaluate.py --config configs/dqn_grid.yaml --n_eval_episodes 20
```

### Test

```bash
pytest
```

---

## 🏗️ Extending the Codebase

### Add a new environment

Subclass `gymnasium.Env` and place it in `src/envs/`.  Register it in `src/envs/__init__.py` and add a YAML config entry with `env: MyNewEnv`.

### Add a new agent

Subclass `src.agents.base_agent.BaseAgent` and implement:
- `select_action(obs, *, deterministic=False)`
- `update(**kwargs) → dict[str, float]`
- `state_dict()` / `load_state_dict(state)`

Add it to `src/agents/__init__.py` and the `_build_agent` helper in `scripts/train.py`.

### Enable TensorBoard

Set `use_tensorboard: true` in your YAML config, then:

```bash
tensorboard --logdir runs/
```

---

## 🔬 Research Directions

- **World Models** – wrap any agent with a learned transition model in `src/models/` to enable imagination-based planning.
- **Multimodal Control** – add image observations via the `CNN` backbone and `FrameStack` wrapper.
- **Human-in-the-Loop** – hook into the `Trainer` `eval_every` callback to surface agent behaviour for human feedback.

