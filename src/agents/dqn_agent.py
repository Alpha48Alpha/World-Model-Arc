"""Deep Q-Network (DQN) agent with experience replay and target network.

Implements the vanilla DQN algorithm from Mnih et al. (2015) with:
  - Epsilon-greedy exploration with linear annealing.
  - Fixed target network updated every ``target_update_freq`` steps.
  - Uniform experience replay buffer.
  - Huber loss for gradient stability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.models.networks import MLP
from src.models.memory import ReplayBuffer


class DQNAgent(BaseAgent):
    """DQN agent for discrete action spaces.

    Parameters
    ----------
    obs_dim:
        Observation dimensionality.
    action_dim:
        Number of discrete actions.
    hidden_sizes:
        MLP hidden layer widths for both online and target networks.
    lr:
        Learning rate.
    gamma:
        Discount factor.
    epsilon_start / epsilon_end / epsilon_decay_steps:
        Linear epsilon-greedy schedule.
    replay_capacity:
        Size of the replay buffer.
    batch_size:
        Mini-batch size for training.
    target_update_freq:
        Number of gradient steps between target network syncs.
    device:
        Torch device string.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_dim, action_dim, device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._update_count = 0

        # Online and target Q-networks
        self.q_net = MLP(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = MLP(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss()

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            obs_dim=obs_dim,
            action_dim=1,
            device=device,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, *, deterministic: bool = False
    ) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        with torch.no_grad():
            q_values = self.q_net(self.obs_to_tensor(obs))
        return int(q_values.argmax(dim=1).item())

    def update(self, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        """Sample a mini-batch and perform one gradient step.

        Returns
        -------
        dict with key ``"loss"`` (0.0 if buffer not yet ready).
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}

        batch = self.replay_buffer.sample(self.batch_size)
        obs = batch["obs"]
        actions = batch["actions"].long().squeeze(1)
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Current Q-values
        q_values = self.q_net(obs)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style: online selects, target evaluates)
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Anneal epsilon linearly
        self._anneal_epsilon()

        return {"loss": float(loss.item())}

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition and increment step counter."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.increment_steps()

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self._total_steps,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state["epsilon"]
        self._total_steps = state["total_steps"]
        self._update_count = state["update_count"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _anneal_epsilon(self) -> None:
        frac = min(1.0, self._total_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = (
            self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        )
