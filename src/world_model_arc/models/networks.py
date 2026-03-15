"""Reusable neural network building blocks.

Provides:
- :class:`MLP` – multi-layer perceptron with configurable depth/width/activation.
- :class:`CNN` – convolutional encoder for grid-based observations.
- :class:`DuelingMLP` – dueling network head (value + advantage streams).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name (case-insensitive)."""
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Fully-connected network with optional layer-norm.

    Args:
        input_dim: Size of the input feature vector.
        output_dim: Size of the output vector.
        hidden_dims: Sizes of intermediate hidden layers.
        activation: Name of the hidden-layer activation function.
        layer_norm: Whether to apply :class:`~torch.nn.LayerNorm` after
            each hidden activation.
        output_activation: Optional activation applied to the output layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "relu",
        layer_norm: bool = False,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        act_fn = get_activation(activation)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(act_fn)
            in_dim = h
            # Re-instantiate so each layer has its own module object
            act_fn = get_activation(activation)

        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------


class CNN(nn.Module):
    """Convolutional encoder for grid/image observations.

    Expects input of shape ``(B, C, H, W)`` and returns a flat feature
    vector of size ``output_dim``.

    Args:
        in_channels: Number of input channels.
        output_dim: Dimensionality of the output embedding.
        channels: Per-layer output channels for each conv block.
        kernel_size: Kernel size used for all conv layers.
        activation: Activation applied after each conv layer.
    """

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        conv_layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            conv_layers.append(
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            conv_layers.append(get_activation(activation))
            conv_layers.append(nn.MaxPool2d(2))
            c_in = c_out

        self.conv = nn.Sequential(*conv_layers)
        self._output_dim = output_dim
        self._c_out = c_in
        # Linear head will be built lazily on first forward pass
        self._head: Optional[nn.Linear] = None

    def _build_head(self, flat_dim: int) -> None:
        self._head = nn.Linear(flat_dim, self._output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        flat = features.flatten(start_dim=1)
        if self._head is None:
            self._build_head(flat.shape[1])
            self._head = self._head.to(x.device)
        return torch.relu(self._head(flat))


# ---------------------------------------------------------------------------
# Dueling MLP
# ---------------------------------------------------------------------------


class DuelingMLP(nn.Module):
    """Dueling network architecture (Wang et al., 2016).

    Splits the penultimate representation into two streams:
    - **Value** stream: V(s) – scalar
    - **Advantage** stream: A(s, a) – per-action scalar

    The Q-values are computed as:
        Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

    Args:
        input_dim: Size of the input observation vector.
        n_actions: Number of discrete actions.
        hidden_dims: Sizes of the shared trunk hidden layers.
        activation: Activation function name.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()

        act = get_activation(activation)
        trunk_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            trunk_layers.extend([nn.Linear(in_dim, h), get_activation(activation)])
            act = get_activation(activation)
            in_dim = h

        self.trunk = nn.Sequential(*trunk_layers)
        self.value_head = nn.Linear(in_dim, 1)
        self.adv_head = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.trunk(x)
        value = self.value_head(h)
        adv = self.adv_head(h)
        return value + adv - adv.mean(dim=-1, keepdim=True)
