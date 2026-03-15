"""World-model stub: a learned transition and reward predictor.

This module provides the scaffolding that downstream researchers can fill
in to build Dyna-style or MBRL agents.  The current implementation is a
simple deterministic MLP transition model trained with supervised regression.

Extension points
----------------
- Replace ``TransitionModel`` with a recurrent / attention-based model.
- Add a reward head and a termination head.
- Implement an ensemble for epistemic uncertainty.
- Add a latent-space encoder for high-dimensional observations (``CNN``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_arc.models.networks import MLP


class WorldModel(nn.Module):
    """Learned environment model: predicts next observation and reward.

    Args:
        obs_dim: Dimensionality of the observation space.
        action_dim: Number of discrete actions.
        hidden_dims: Hidden layer sizes for both sub-networks.
        activation: Activation function name.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        in_dim = obs_dim + action_dim  # action embedded as one-hot

        self.transition_net = MLP(
            input_dim=in_dim,
            output_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.reward_net = MLP(
            input_dim=in_dim,
            output_dim=1,
            hidden_dims=hidden_dims[:1],
            activation=activation,
        )
        self._obs_dim = obs_dim
        self._action_dim = action_dim

    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next observation and reward.

        Args:
            obs: Current observations, shape ``(B, obs_dim)``.
            action: Discrete action indices, shape ``(B,)``.

        Returns:
            Tuple of:
            - ``next_obs`` – predicted next observation ``(B, obs_dim)``.
            - ``reward`` – predicted reward ``(B, 1)``.
        """
        one_hot = F.one_hot(action.long(), num_classes=self._action_dim).float()
        x = torch.cat([obs, one_hot], dim=-1)
        next_obs = obs + self.transition_net(x)  # residual connection
        reward = self.reward_net(x)
        return next_obs, reward

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised regression loss for fitting the world model.

        Args:
            obs: Batch of observations ``(B, obs_dim)``.
            action: Batch of action indices ``(B,)``.
            next_obs: Batch of true next observations ``(B, obs_dim)``.
            reward: Batch of true rewards ``(B,)`` or ``(B, 1)``.

        Returns:
            Scalar loss tensor.
        """
        pred_next, pred_reward = self(obs, action)
        trans_loss = F.mse_loss(pred_next, next_obs)
        rew_loss = F.mse_loss(pred_reward.squeeze(-1), reward.float().squeeze(-1))
        return trans_loss + rew_loss
