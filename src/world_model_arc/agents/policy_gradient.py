"""REINFORCE (Monte-Carlo Policy Gradient) agent.

Implements vanilla policy gradient with:
- Parameterised softmax policy (MLP)
- Optional entropy regularisation
- Optional baseline (mean return)
- Gradient clipping

Reference:
    Williams, R. J., "Simple statistical gradient-following algorithms
    for connectionist reinforcement learning", *Machine Learning*, 1992.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_arc.agents.base import BaseAgent
from world_model_arc.models.networks import MLP


class PolicyGradientAgent(BaseAgent):
    """REINFORCE agent with optional entropy regularisation.

    Transitions are collected in an episode buffer and used to update
    the policy at the end of each episode via :meth:`end_episode`.

    Args:
        obs_shape: Shape of a flat observation vector.
        n_actions: Number of discrete actions.
        lr: Learning rate.
        gamma: Discount factor.
        entropy_coef: Weight of the entropy bonus term.
        hidden_dims: MLP hidden layer sizes.
        activation: Hidden-layer activation function name.
        clip_grad: Max gradient norm (0 = disabled).
        device: Torch device string.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        clip_grad: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(obs_shape, n_actions, device)

        assert len(obs_shape) == 1, "PolicyGradientAgent requires a flat observation vector."
        obs_dim = obs_shape[0]

        self.policy_net = MLP(
            input_dim=obs_dim,
            output_dim=n_actions,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.clip_grad = clip_grad

        # Episode buffer
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._entropies: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Sample from the policy distribution.

        Args:
            obs: Current observation.
            deterministic: If *True*, take the argmax action (no sampling).

        Returns:
            Integer action index.
        """
        obs_t = self._obs_to_tensor(obs)
        logits = self.policy_net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = int(logits.argmax(dim=-1).item())
        else:
            action = int(dist.sample().item())
            # Store for end-of-episode update
            self._log_probs.append(dist.log_prob(torch.tensor(action, device=self.device)))
            self._entropies.append(dist.entropy())

        return action

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Accumulate reward for the current episode."""
        if not done or len(self._log_probs) == len(self._rewards) + 1:
            # Only append reward if a log_prob was recorded (i.e. non-deterministic)
            if len(self._log_probs) > len(self._rewards):
                self._rewards.append(reward)

    def update(self) -> dict[str, float]:
        """Policy gradient update (called via :meth:`end_episode`)."""
        return {}

    def end_episode(self) -> dict[str, float]:
        """Compute returns and run a REINFORCE gradient update.

        Called automatically by the :class:`~world_model_arc.training.trainer.Trainer`
        at the end of every episode.
        """
        if not self._log_probs:
            return {}

        # Compute discounted returns
        returns: list[float] = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalise returns (baseline)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(self._log_probs)
        entropies_t = torch.stack(self._entropies)

        # Truncate to the minimum length in case of misalignment
        n = min(len(log_probs_t), len(returns_t))
        pg_loss = -(log_probs_t[:n] * returns_t[:n]).mean()
        entropy_loss = -self.entropy_coef * entropies_t[:n].mean()
        loss = pg_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
        self.optimizer.step()

        self._train_steps += 1
        metrics = {
            "pg_loss": float(pg_loss.item()),
            "entropy": float(entropies_t.mean().item()),
            "loss": float(loss.item()),
        }

        # Clear episode buffer
        self._log_probs.clear()
        self._rewards.clear()
        self._entropies.clear()

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy_net.load_state_dict(state["policy_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._train_steps = state.get("train_steps", 0)
