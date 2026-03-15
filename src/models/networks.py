"""Reusable PyTorch neural network building blocks.

Provides:
  - MLP            : configurable multi-layer perceptron
  - CNN            : configurable 1-D convolutional feature extractor
  - ActorCritic    : shared-backbone actor-critic head for PPO / A2C
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: Optional[type[nn.Module]] = None,
) -> nn.Sequential:
    """Build a sequential MLP from layer size specifications."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """General-purpose multi-layer perceptron.

    Parameters
    ----------
    in_dim:
        Dimensionality of input features.
    out_dim:
        Dimensionality of output.
    hidden_sizes:
        Sequence of hidden layer widths.
    activation:
        Hidden-layer activation class.
    output_activation:
        Optional output-layer activation class.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.net = _make_mlp(
            in_dim, hidden_sizes, out_dim, activation, output_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """1-D convolutional feature extractor followed by a linear projection.

    Intended for sequence / vector observations.  Input shape:
    ``(batch, in_dim)`` is reshaped to ``(batch, 1, in_dim)`` before
    passing through the convolutional layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        channels: Sequence[int] = (32, 64),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        conv_layers: list[nn.Module] = []
        in_ch = 1
        length = in_dim
        for out_ch in channels:
            conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=1)
            )
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        self.convs = nn.Sequential(*conv_layers)

        # Compute flattened size after convolutions
        dummy = torch.zeros(1, 1, in_dim)
        flat_size = self.convs(dummy).flatten(1).shape[1]
        self.proj = nn.Linear(flat_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        x = x.unsqueeze(1)          # -> (batch, 1, in_dim)
        x = self.convs(x)           # -> (batch, C, L)
        x = x.flatten(1)            # -> (batch, C*L)
        return F.relu(self.proj(x)) # -> (batch, out_dim)


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network for PPO / A2C / REINFORCE.

    A common trunk MLP is followed by two heads:
      - *actor*:  outputs un-normalised action logits (discrete) or mean
                  (continuous).
      - *critic*: outputs a scalar state-value estimate V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        continuous: bool = False,
    ) -> None:
        super().__init__()
        self.continuous = continuous

        # Shared trunk
        self.trunk = _make_mlp(
            in_dim=obs_dim,
            hidden_sizes=hidden_sizes[:-1] if len(hidden_sizes) > 1 else hidden_sizes,
            out_dim=hidden_sizes[-1],
            activation=nn.Tanh,
        )

        # Actor head
        self.actor_head = nn.Linear(hidden_sizes[-1], action_dim)

        # Critic head
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

        if continuous:
            # Log-std parameter (state-independent)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain for actor head to keep initial policy close to uniform
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits_or_mean, value).

        For continuous policies, returns the Gaussian mean and the
        log-std is available via ``self.log_std``.
        """
        features = self.trunk(obs)
        logits_or_mean = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits_or_mean, value
