"""Tests for the environment implementations."""

from __future__ import annotations

import numpy as np
import pytest

from world_model_arc.envs import ContinuousWorld, GridWorld
from world_model_arc.envs.base import StepResult


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------


class TestGridWorld:
    def test_reset_returns_observation(self):
        env = GridWorld(height=4, width=4)
        obs = env.reset(seed=0)
        assert obs.shape == env.observation_shape
        assert obs.dtype == np.float32

    def test_step_returns_step_result(self):
        env = GridWorld(height=4, width=4)
        env.reset(seed=0)
        result = env.step(0)  # move up
        assert isinstance(result, StepResult)
        assert result.observation.shape == env.observation_shape
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)

    def test_flat_observation_shape(self):
        env = GridWorld(height=6, width=6, obs_mode="flat")
        obs = env.reset(seed=1)
        assert obs.shape == (3,)

    def test_onehot_observation_shape(self):
        env = GridWorld(height=4, width=4, obs_mode="onehot")
        obs = env.reset(seed=1)
        # (n_tile_types, H, W)
        from world_model_arc.envs.gridworld import Tile
        assert obs.shape == (len(Tile), 4, 4)

    def test_goal_reached_terminates_episode(self):
        # Place agent right next to goal; step towards it
        env = GridWorld(
            height=3,
            width=3,
            goals=[(2, 2)],
            hazards=[],
            walls=[],
            start=(2, 1),
            max_steps=100,
        )
        env.reset(seed=0)
        # action=1 → right → (2,2) = goal
        result = env.step(1)
        assert result.info.get("goal_reached"), "Goal should be reached"
        assert result.done

    def test_hazard_terminates_episode(self):
        env = GridWorld(
            height=3,
            width=3,
            goals=[(2, 2)],
            hazards=[(1, 0)],
            walls=[],
            start=(0, 0),
            max_steps=100,
        )
        env.reset(seed=0)
        # action=2 → down → (1,0) = hazard
        result = env.step(2)
        assert result.info.get("hazard_hit"), "Hazard should be hit"
        assert result.done

    def test_wall_blocks_movement(self):
        env = GridWorld(
            height=4,
            width=4,
            goals=[(3, 3)],
            hazards=[],
            walls=[(0, 1)],
            start=(0, 0),
            max_steps=50,
        )
        env.reset(seed=0)
        # action=1 → right → (0,1) which is wall → should stay at (0,0)
        result = env.step(1)
        r, c = env._agent_pos
        assert (r, c) == (0, 0), f"Agent should bounce back from wall, got ({r},{c})"

    def test_max_steps_truncates(self):
        env = GridWorld(height=4, width=4, goals=[(3, 3)], max_steps=3)
        env.reset(seed=42)
        for _ in range(3):
            result = env.step(0)
        assert result.truncated or result.done

    def test_render_returns_rgb(self):
        env = GridWorld(height=4, width=4)
        env.reset(seed=0)
        img = env.render()
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    def test_slip_probability(self):
        """With slip_prob=1.0, actions are always random."""
        env = GridWorld(height=4, width=4, slip_prob=1.0)
        env.reset(seed=99)
        # Just verify it doesn't raise
        for a in range(4):
            env.reset(seed=99)
            result = env.step(a)
            assert result.observation.shape == env.observation_shape

    def test_action_size(self):
        env = GridWorld()
        assert env.action_size == 4

    def test_multiple_resets_are_independent(self):
        env = GridWorld(height=4, width=4, start=None)
        obs1 = env.reset(seed=0)
        obs2 = env.reset(seed=1)
        # Different seeds may produce different starting positions
        # At minimum both should be valid observations
        assert obs1.shape == env.observation_shape
        assert obs2.shape == env.observation_shape


# ---------------------------------------------------------------------------
# ContinuousWorld
# ---------------------------------------------------------------------------


class TestContinuousWorld:
    def test_reset_returns_observation(self):
        env = ContinuousWorld()
        obs = env.reset(seed=0)
        assert obs.shape == env.observation_shape
        assert obs.dtype == np.float32

    def test_step_returns_step_result(self):
        env = ContinuousWorld()
        env.reset(seed=0)
        result = env.step(0)  # no-op
        assert isinstance(result, StepResult)
        assert result.observation.shape == env.observation_shape

    def test_action_size(self):
        env = ContinuousWorld()
        assert env.action_size == 5

    def test_observation_shape(self):
        env = ContinuousWorld()
        assert env.observation_shape == (7,)

    def test_max_steps_truncates(self):
        env = ContinuousWorld(max_steps=5)
        env.reset(seed=0)
        for _ in range(5):
            result = env.step(0)
        assert result.truncated or result.done

    def test_render_returns_rgb(self):
        env = ContinuousWorld()
        env.reset(seed=0)
        img = env.render()
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_position_stays_in_bounds(self):
        env = ContinuousWorld(max_steps=200)
        env.reset(seed=7)
        for _ in range(50):
            action = np.random.randint(0, 5)
            result = env.step(action)
            obs = result.observation
            assert 0.0 <= obs[0] <= 1.0, "x out of bounds"
            assert 0.0 <= obs[1] <= 1.0, "y out of bounds"
            if result.done or result.truncated:
                env.reset(seed=7)
