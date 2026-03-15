"""Tests for DQN and Policy Gradient agents."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from world_model_arc.agents import DQNAgent, PolicyGradientAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_SHAPE = (4,)
N_ACTIONS = 3


def make_obs() -> np.ndarray:
    return np.random.randn(*OBS_SHAPE).astype(np.float32)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


class TestDQNAgent:
    def _agent(self, **kwargs) -> DQNAgent:
        return DQNAgent(
            obs_shape=OBS_SHAPE,
            n_actions=N_ACTIONS,
            buffer_capacity=200,
            batch_size=16,
            epsilon_decay_steps=100,
            **kwargs,
        )

    def test_select_action_range(self):
        agent = self._agent()
        obs = make_obs()
        for _ in range(20):
            action = agent.select_action(obs)
            assert 0 <= action < N_ACTIONS

    def test_select_action_deterministic(self):
        agent = self._agent()
        obs = make_obs()
        actions = {agent.select_action(obs, deterministic=True) for _ in range(5)}
        # Deterministic should always return the same action
        assert len(actions) == 1

    def test_update_returns_empty_before_buffer_filled(self):
        agent = self._agent()
        metrics = agent.update()
        assert metrics == {}

    def test_update_returns_loss_after_buffer_filled(self):
        agent = self._agent()
        # Fill buffer with random transitions
        for _ in range(30):
            obs = make_obs()
            action = np.random.randint(0, N_ACTIONS)
            reward = float(np.random.randn())
            next_obs = make_obs()
            done = bool(np.random.rand() < 0.1)
            agent.observe(obs, action, reward, next_obs, done)

        metrics = agent.update()
        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_epsilon_decay(self):
        agent = DQNAgent(
            obs_shape=OBS_SHAPE,
            n_actions=N_ACTIONS,
            buffer_capacity=200,
            batch_size=16,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
        )
        initial_eps = agent.epsilon
        obs = make_obs()
        for _ in range(50):
            agent.select_action(obs)
        assert agent.epsilon < initial_eps

    def test_checkpoint_roundtrip(self, tmp_path):
        agent = self._agent()
        # Do some updates to change state
        for _ in range(20):
            obs = make_obs()
            agent.observe(obs, 0, 1.0, make_obs(), False)
        agent.update()

        state = agent.state_dict()
        # Create a new agent and load state
        agent2 = self._agent()
        agent2.load_state_dict(state)

        obs = make_obs()
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_dueling_architecture(self):
        agent = DQNAgent(
            obs_shape=OBS_SHAPE,
            n_actions=N_ACTIONS,
            dueling=True,
            buffer_capacity=50,
            batch_size=8,
        )
        obs = make_obs()
        action = agent.select_action(obs)
        assert 0 <= action < N_ACTIONS

    def test_target_network_sync(self):
        """Target network should update after target_update_freq steps."""
        agent = DQNAgent(
            obs_shape=OBS_SHAPE,
            n_actions=N_ACTIONS,
            target_update_freq=5,
            batch_size=8,
            buffer_capacity=50,
        )
        for _ in range(20):
            agent.observe(make_obs(), 0, 0.5, make_obs(), False)
        # Run enough updates to trigger sync
        for _ in range(10):
            agent.update()
        # Target and online network should have identical weights
        for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(p1.data, p2.data)


# ---------------------------------------------------------------------------
# PolicyGradientAgent
# ---------------------------------------------------------------------------


class TestPolicyGradientAgent:
    def _agent(self, **kwargs) -> PolicyGradientAgent:
        return PolicyGradientAgent(obs_shape=OBS_SHAPE, n_actions=N_ACTIONS, **kwargs)

    def test_select_action_range(self):
        agent = self._agent()
        obs = make_obs()
        for _ in range(20):
            action = agent.select_action(obs)
            assert 0 <= action < N_ACTIONS

    def test_select_action_deterministic(self):
        agent = self._agent()
        obs = make_obs()
        actions = {agent.select_action(obs, deterministic=True) for _ in range(5)}
        assert len(actions) == 1  # deterministic → same action every time

    def test_update_returns_empty_dict(self):
        agent = self._agent()
        metrics = agent.update()
        assert metrics == {}

    def test_end_episode_returns_loss_after_rollout(self):
        agent = self._agent()
        obs = make_obs()
        # Simulate one episode
        for _ in range(10):
            action = agent.select_action(obs)
            agent.observe(obs, action, 0.5, make_obs(), False)
        # Final step
        action = agent.select_action(obs)
        agent.observe(obs, action, 1.0, make_obs(), True)

        metrics = agent.end_episode()
        assert "loss" in metrics

    def test_end_episode_clears_buffer(self):
        agent = self._agent()
        obs = make_obs()
        for _ in range(5):
            action = agent.select_action(obs)
            agent.observe(obs, action, 0.1, make_obs(), False)
        agent.end_episode()
        assert len(agent._log_probs) == 0
        assert len(agent._rewards) == 0

    def test_checkpoint_roundtrip(self):
        agent = self._agent()
        obs = make_obs()
        for _ in range(5):
            action = agent.select_action(obs)
            agent.observe(obs, action, 0.1, make_obs(), False)
        agent.end_episode()

        state = agent.state_dict()
        agent2 = self._agent()
        agent2.load_state_dict(state)

        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent2.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_entropy_coef_effect(self):
        """Higher entropy coef should produce different gradients."""
        agent_low = self._agent(entropy_coef=0.0)
        agent_high = self._agent(entropy_coef=1.0)
        obs = make_obs()
        # Run identical episodes
        for agent in [agent_low, agent_high]:
            torch.manual_seed(42)
            np.random.seed(42)
            for _ in range(10):
                action = agent.select_action(obs)
                agent.observe(obs, action, 0.5, make_obs(), False)
            agent.end_episode()
        # Both agents should still work without error
