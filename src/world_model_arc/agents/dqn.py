"""Deep Q-Network (DQN) agent.

Implements:
- ε-greedy exploration with linear decay
- Experience replay with a circular buffer
- Target network with periodic hard updates
- Optional dueling network architecture (Dueling DQN)
- Gradient clipping

Reference:
    Mnih et al., "Human-level control through deep reinforcement learning",
    *Nature*, 2015.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_arc.agents.base import BaseAgent
from world_model_arc.models.networks import DuelingMLP, MLP
from world_model_arc.utils.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    """DQN agent with replay buffer and target network.

    Args:
        obs_shape: Shape of a flat observation vector.
        n_actions: Number of discrete actions.
        lr: Learning rate for the Adam optimizer.
        gamma: Discount factor.
        epsilon_start: Initial exploration probability.
        epsilon_end: Minimum exploration probability.
        epsilon_decay_steps: Number of *select_action* calls over which ε
            is linearly annealed from *epsilon_start* to *epsilon_end*.
        target_update_freq: How many :meth:`update` calls between hard
            target-network synchronisation.
        buffer_capacity: Maximum size of the replay buffer.
        batch_size: Mini-batch size for each gradient update.
        hidden_dims: Sizes of MLP hidden layers.
        activation: Hidden-layer activation function name.
        dueling: Use the dueling architecture.
        clip_grad: Max gradient norm (0 = disabled).
        device: Torch device string.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_freq: int = 1_000,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        dueling: bool = False,
        clip_grad: float = 10.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_shape, n_actions, device)

        assert len(obs_shape) == 1, "DQNAgent currently requires a flat observation vector."
        obs_dim = obs_shape[0]

        # Networks
        net_cls = DuelingMLP if dueling else MLP
        if dueling:
            self.q_net: nn.Module = DuelingMLP(
                obs_dim, n_actions, hidden_dims=hidden_dims, activation=activation
            ).to(self.device)
            self.target_net: nn.Module = DuelingMLP(
                obs_dim, n_actions, hidden_dims=hidden_dims, activation=activation
            ).to(self.device)
        else:
            self.q_net = MLP(
                obs_dim, n_actions, hidden_dims=hidden_dims, activation=activation
            ).to(self.device)
            self.target_net = MLP(
                obs_dim, n_actions, hidden_dims=hidden_dims, activation=activation
            ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, obs_shape, device=device)
        self.batch_size = batch_size

        # Hyper-parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.clip_grad = clip_grad

        self._action_steps = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """ε-greedy action selection.

        Args:
            obs: Current observation.
            deterministic: If *True*, always take the greedy action.

        Returns:
            Integer action index.
        """
        if not deterministic and np.random.random() < self.epsilon:
            action = int(np.random.randint(0, self.n_actions))
        else:
            with torch.no_grad():
                q_vals = self.q_net(self._obs_to_tensor(obs))
            action = int(q_vals.argmax(dim=-1).item())

        # Anneal epsilon
        if not deterministic:
            self._action_steps += 1
            frac = min(1.0, self._action_steps / max(1, self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        return action

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> dict[str, float]:
        """Sample a mini-batch and perform one gradient step.

        Returns an empty dict if the buffer is not large enough yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_q = self.target_net(batch.next_obs)
            max_next_q = next_q.max(dim=-1).values
            target = batch.reward + self.gamma * max_next_q * (1.0 - batch.done)

        current_q = self.q_net(batch.obs)
        q_chosen = current_q.gather(1, batch.action.unsqueeze(-1)).squeeze(-1)

        loss = F.smooth_l1_loss(q_chosen, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_grad)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self._train_steps,
            "action_steps": self._action_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state.get("epsilon", self.epsilon)
        self._train_steps = state.get("train_steps", 0)
        self._action_steps = state.get("action_steps", 0)
