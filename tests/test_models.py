"""Tests for neural network architectures."""

from __future__ import annotations

import pytest
import torch

from world_model_arc.models.networks import CNN, MLP, DuelingMLP


class TestMLP:
    def test_forward_shape(self):
        net = MLP(input_dim=8, output_dim=4, hidden_dims=(32, 32))
        x = torch.randn(16, 8)
        out = net(x)
        assert out.shape == (16, 4)

    def test_single_hidden(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=(16,))
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 2)

    def test_no_hidden(self):
        net = MLP(input_dim=4, output_dim=2, hidden_dims=())
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 2)

    def test_layer_norm(self):
        net = MLP(input_dim=8, output_dim=4, hidden_dims=(32,), layer_norm=True)
        x = torch.randn(16, 8)
        out = net(x)
        assert out.shape == (16, 4)

    def test_activations(self):
        for act in ["relu", "tanh", "elu", "leaky_relu"]:
            net = MLP(input_dim=4, output_dim=2, activation=act)
            out = net(torch.randn(8, 4))
            assert out.shape == (8, 2)

    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            MLP(input_dim=4, output_dim=2, activation="invalid")


class TestDuelingMLP:
    def test_forward_shape(self):
        net = DuelingMLP(input_dim=8, n_actions=4, hidden_dims=(32, 32))
        x = torch.randn(16, 8)
        out = net(x)
        assert out.shape == (16, 4)

    def test_q_values_are_finite(self):
        net = DuelingMLP(input_dim=8, n_actions=4)
        x = torch.randn(32, 8)
        out = net(x)
        assert torch.isfinite(out).all()


class TestCNN:
    def test_forward_shape(self):
        net = CNN(in_channels=3, output_dim=64, channels=(16, 32))
        x = torch.randn(4, 3, 16, 16)
        out = net(x)
        assert out.shape == (4, 64)

    def test_single_channel(self):
        net = CNN(in_channels=1, output_dim=32, channels=(8,))
        x = torch.randn(2, 1, 8, 8)
        out = net(x)
        assert out.shape == (2, 32)
