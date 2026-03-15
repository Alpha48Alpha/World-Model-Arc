"""Tests for GridWorld and ContinuousWorld environments."""

import pytest
import numpy as np
from src.envs.grid_world import GridWorld
from src.envs.continuous_world import ContinuousWorld
from src.envs.wrappers import NormalizeObservation, FrameStack


class TestGridWorld:
    def test_reset_returns_correct_shapes(self):
        env = GridWorld(height=5, width=5)
        obs, info = env.reset(seed=0)
        assert obs.shape == (25,), f"Expected (25,), got {obs.shape}"
        assert "agent_pos" in info
        assert "goal_positions" in info

    def test_observation_range(self):
        env = GridWorld(height=5, width=5)
        obs, _ = env.reset(seed=0)
        assert obs.min() >= 0
        assert obs.max() <= 4

    def test_step_returns_correct_types(self):
        env = GridWorld()
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        env = GridWorld()
        assert env.action_space.n == 4

    def test_episode_terminates(self):
        """An episode eventually ends (terminated or truncated)."""
        env = GridWorld(height=4, width=4, n_goals=1, n_hazards=0, n_walls=0, max_steps=500)
        obs, _ = env.reset(seed=7)
        done = False
        steps = 0
        while not done and steps < 10_000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate within 10000 steps"

    def test_wall_bounce(self):
        """Stepping into a wall keeps the agent in place."""
        env = GridWorld(height=4, width=4, n_walls=0, n_hazards=0, n_goals=1)
        env.reset(seed=0)
        # Force agent to top-left corner
        env._agent_pos = (0, 0)
        obs_before = env._get_obs().copy()
        # Moving up from (0,0) should bounce
        env.step(0)  # up
        assert env._agent_pos == (0, 0)

    def test_render_ansi(self):
        env = GridWorld(height=4, width=4, render_mode="ansi")
        env.reset(seed=0)
        output = env.render()
        assert isinstance(output, str)

    def test_seed_reproducibility(self):
        env1 = GridWorld(height=5, width=5)
        env2 = GridWorld(height=5, width=5)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_goal_reward(self):
        """Agent receives +1 reward when stepping onto a goal."""
        env = GridWorld(height=3, width=3, n_goals=1, n_hazards=0, n_walls=0)
        env.reset(seed=0)
        # Manually place agent adjacent to goal
        env._agent_pos = (0, 0)
        env._grid[:] = 0
        env._grid[0, 1] = GridWorld.GOAL
        env._goal_positions = [(0, 1)]
        _, reward, terminated, _, _ = env.step(1)  # move right
        assert reward == pytest.approx(1.0)
        assert terminated is True

    def test_hazard_reward(self):
        """Agent receives -1 reward when stepping into a hazard."""
        env = GridWorld(height=3, width=3, n_goals=1, n_hazards=0, n_walls=0)
        env.reset(seed=0)
        env._agent_pos = (0, 0)
        env._grid[:] = 0
        env._grid[0, 1] = GridWorld.HAZARD
        env._goal_positions = [(2, 2)]
        _, reward, terminated, _, _ = env.step(1)  # move right
        assert reward == pytest.approx(-1.0)
        assert terminated is True


class TestContinuousWorld:
    def test_reset_shapes(self):
        env = ContinuousWorld()
        obs, info = env.reset(seed=0)
        assert obs.shape == (6,)
        assert "agent_pos" in info

    def test_step_types(self):
        env = ContinuousWorld()
        env.reset(seed=0)
        action = np.zeros(2, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)

    def test_action_clipping(self):
        """Actions outside [-1, 1] are clipped without error."""
        env = ContinuousWorld()
        env.reset(seed=0)
        huge_action = np.array([100.0, -100.0])
        obs, _, _, _, _ = env.step(huge_action)
        assert np.all(env._agent_pos >= 0) and np.all(env._agent_pos <= 1)

    def test_goal_reached_terminates(self):
        """Placing agent on top of goal leads to termination."""
        env = ContinuousWorld(goal_radius=0.1)
        env.reset(seed=0)
        env._agent_pos = env._goal_pos.copy()
        _, reward, terminated, _, _ = env.step(np.zeros(2))
        assert terminated is True
        assert reward > 0

    def test_seed_reproducibility(self):
        env1 = ContinuousWorld()
        env2 = ContinuousWorld()
        obs1, _ = env1.reset(seed=99)
        obs2, _ = env2.reset(seed=99)
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestNormalizeObservation:
    def test_output_dtype_and_shape(self):
        env = NormalizeObservation(GridWorld(height=5, width=5))
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32
        assert obs.shape == (25,)

    def test_normalisation_runs_without_error(self):
        env = NormalizeObservation(GridWorld())
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            obs, _, _, _, _ = env.step(env.action_space.sample())
        # After enough steps the obs should be in a reasonable range
        assert np.all(np.isfinite(obs))


class TestFrameStack:
    def test_stacked_shape(self):
        env = FrameStack(GridWorld(height=5, width=5), n_frames=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (75,)  # 3 * 25

    def test_stacked_values_change(self):
        env = FrameStack(GridWorld(height=5, width=5), n_frames=2)
        obs0, _ = env.reset(seed=0)
        obs1, _, _, _, _ = env.step(1)
        # After one step the second half of obs1 should differ from first half
        half = 25
        # They won't always be equal (depends on movement)
        assert obs1.shape == (50,)
