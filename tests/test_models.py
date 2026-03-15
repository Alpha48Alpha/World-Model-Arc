"""Tests for neural network models and memory buffers."""

import pytest
import numpy as np
import torch

from src.models.networks import MLP, CNN, ActorCritic
from src.models.memory import ReplayBuffer, RolloutStorage


class TestMLP:
    def test_forward_shape(self):
        net = MLP(in_dim=16, out_dim=4, hidden_sizes=(32, 32))
        x = torch.randn(8, 16)
        out = net(x)
        assert out.shape == (8, 4)

    def test_single_sample(self):
        net = MLP(in_dim=10, out_dim=3)
        x = torch.randn(1, 10)
        out = net(x)
        assert out.shape == (1, 3)

    def test_custom_activation(self):
        net = MLP(in_dim=8, out_dim=2, hidden_sizes=(16,), activation=torch.nn.Tanh)
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 2)

    def test_gradient_flows(self):
        net = MLP(in_dim=8, out_dim=2)
        x = torch.randn(4, 8)
        loss = net(x).sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None


class TestCNN:
    def test_forward_shape(self):
        net = CNN(in_dim=64, out_dim=16)
        x = torch.randn(4, 64)
        out = net(x)
        assert out.shape == (4, 16)

    def test_gradient_flows(self):
        net = CNN(in_dim=32, out_dim=8)
        x = torch.randn(2, 32)
        loss = net(x).sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None


class TestActorCritic:
    def test_output_shapes_discrete(self):
        ac = ActorCritic(obs_dim=16, action_dim=4)
        x = torch.randn(3, 16)
        logits, value = ac(x)
        assert logits.shape == (3, 4)
        assert value.shape == (3,)

    def test_value_is_scalar_per_sample(self):
        ac = ActorCritic(obs_dim=8, action_dim=3)
        x = torch.randn(5, 8)
        _, value = ac(x)
        assert value.dim() == 1
        assert value.shape[0] == 5

    def test_gradient_flows(self):
        ac = ActorCritic(obs_dim=8, action_dim=4)
        x = torch.randn(2, 8)
        logits, value = ac(x)
        (logits.sum() + value.sum()).backward()
        for p in ac.parameters():
            assert p.grad is not None


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=100, obs_dim=4, action_dim=1)
        for i in range(20):
            obs = np.random.randn(4).astype(np.float32)
            nobs = np.random.randn(4).astype(np.float32)
            buf.add(obs, i % 3, float(i), nobs, i % 5 == 0)
        assert len(buf) == 20
        batch = buf.sample(8)
        assert batch["obs"].shape == (8, 4)
        assert batch["rewards"].shape == (8,)
        assert batch["dones"].shape == (8,)

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=10, obs_dim=4)
        for i in range(25):
            obs = np.zeros(4, dtype=np.float32)
            buf.add(obs, 0, 0.0, obs, False)
        assert len(buf) == 10  # Circular buffer

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=100, obs_dim=4)
        assert not buf.is_ready
        buf.add(np.zeros(4, dtype=np.float32), 0, 0.0, np.zeros(4, dtype=np.float32), False)
        assert buf.is_ready

    def test_tensor_device(self):
        buf = ReplayBuffer(capacity=50, obs_dim=4, device="cpu")
        obs = np.ones(4, dtype=np.float32)
        buf.add(obs, 1, 0.5, obs, False)
        batch = buf.sample(1)
        assert batch["obs"].device.type == "cpu"


class TestRolloutStorage:
    def _fill(self, storage, n=10):
        for i in range(n):
            obs = np.random.randn(storage.obs_dim).astype(np.float32)
            storage.add(obs, action=i % 3, reward=float(i), log_prob=-0.5,
                        value=float(i * 0.1), done=(i == n - 1))

    def test_add_and_length(self):
        rs = RolloutStorage(obs_dim=8, max_steps=50)
        self._fill(rs, 10)
        assert len(rs) == 10

    def test_reset_clears(self):
        rs = RolloutStorage(obs_dim=8, max_steps=50)
        self._fill(rs, 10)
        rs.reset()
        assert len(rs) == 0

    def test_compute_returns_shape(self):
        rs = RolloutStorage(obs_dim=8, max_steps=50)
        self._fill(rs, 10)
        returns = rs.compute_returns(gamma=0.99)
        assert returns.shape == (10,)
        assert np.all(np.isfinite(returns))

    def test_compute_gae_shapes(self):
        rs = RolloutStorage(obs_dim=8, max_steps=50)
        self._fill(rs, 10)
        adv, rets = rs.compute_gae(gamma=0.99, lam=0.95)
        assert adv.shape == (10,)
        assert rets.shape == (10,)

    def test_returns_discounted_correctly(self):
        """Simple 3-step trajectory: rewards [1, 1, 1], gamma=0.5."""
        rs = RolloutStorage(obs_dim=2, max_steps=10)
        obs = np.zeros(2, dtype=np.float32)
        rs.add(obs, 0, 1.0, 0.0, 0.0, False)
        rs.add(obs, 0, 1.0, 0.0, 0.0, False)
        rs.add(obs, 0, 1.0, 0.0, 0.0, True)
        returns = rs.compute_returns(gamma=0.5)
        # Expected: [1 + 0.5*(1 + 0.5*1), 1 + 0.5*1, 1] = [1.75, 1.5, 1.0]
        np.testing.assert_allclose(returns, [1.75, 1.5, 1.0], rtol=1e-5)

    def test_overflow_raises(self):
        rs = RolloutStorage(obs_dim=2, max_steps=2)
        obs = np.zeros(2, dtype=np.float32)
        rs.add(obs, 0, 0.0, 0.0, 0.0, False)
        rs.add(obs, 0, 0.0, 0.0, 0.0, False)
        with pytest.raises(RuntimeError):
            rs.add(obs, 0, 0.0, 0.0, 0.0, False)

    def test_as_tensors(self):
        rs = RolloutStorage(obs_dim=4, max_steps=20)
        self._fill(rs, 5)
        tensors = rs.as_tensors()
        assert "obs" in tensors
        assert tensors["obs"].shape == (5, 4)
        assert isinstance(tensors["obs"], torch.Tensor)
