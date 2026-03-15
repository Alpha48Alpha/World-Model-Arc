"""Continuous-state simulated world.

A 2-D continuous navigation environment where the agent controls velocity
components to reach a goal region while avoiding circular hazard zones.
Follows the Gymnasium API.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousWorld(gym.Env):
    """2-D continuous navigation task.

    The agent is a point mass in [0, 1]^2 that applies (dx, dy) velocity
    commands clamped to [-max_speed, max_speed].

    Observation (6-D float32):
        [agent_x, agent_y, goal_x, goal_y, dist_to_goal, step_frac]

    Action (2-D continuous float32):
        [dx, dy] in [-1, 1]

    Reward:
        Dense: -dist_to_goal + 1.0 bonus on success.
        Penalty: -0.5 on hazard collision.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_speed: float = 0.05,
        goal_radius: float = 0.07,
        n_hazards: int = 3,
        hazard_radius: float = 0.08,
        max_steps: int = 300,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.n_hazards = n_hazards
        self.hazard_radius = hazard_radius
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: [ax, ay, gx, gy, dist, step_frac]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        # Action: continuous velocity command
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._agent_pos: np.ndarray = np.zeros(2)
        self._goal_pos: np.ndarray = np.zeros(2)
        self._hazard_positions: list[np.ndarray] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0

        # Sample positions ensuring agent and goal are far enough apart
        while True:
            self._agent_pos = self._rng.uniform(0.1, 0.9, size=2).astype(np.float32)
            self._goal_pos = self._rng.uniform(0.1, 0.9, size=2).astype(np.float32)
            if np.linalg.norm(self._agent_pos - self._goal_pos) > 0.3:
                break

        # Sample hazard positions (not overlapping goal or agent)
        self._hazard_positions = []
        for _ in range(self.n_hazards):
            for _ in range(50):  # rejection sampling
                hp = self._rng.uniform(0.1, 0.9, size=2).astype(np.float32)
                if (
                    np.linalg.norm(hp - self._agent_pos) > 0.15
                    and np.linalg.norm(hp - self._goal_pos) > 0.15
                ):
                    self._hazard_positions.append(hp)
                    break

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        velocity = action * self.max_speed
        self._agent_pos = np.clip(self._agent_pos + velocity, 0.0, 1.0).astype(
            np.float32
        )

        dist = float(np.linalg.norm(self._agent_pos - self._goal_pos))
        reward = -dist
        terminated = False

        if dist < self.goal_radius:
            reward += 1.0
            terminated = True
        else:
            for hp in self._hazard_positions:
                if np.linalg.norm(self._agent_pos - hp) < self.hazard_radius:
                    reward -= 0.5
                    break

        truncated = (not terminated) and (self._step_count >= self.max_steps)
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        msg = (
            f"Step {self._step_count:4d} | "
            f"Agent ({self._agent_pos[0]:.2f}, {self._agent_pos[1]:.2f}) | "
            f"Goal ({self._goal_pos[0]:.2f}, {self._goal_pos[1]:.2f})"
        )
        print(msg)
        return msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        dist = float(np.linalg.norm(self._agent_pos - self._goal_pos))
        step_frac = self._step_count / self.max_steps
        return np.array(
            [
                self._agent_pos[0],
                self._agent_pos[1],
                self._goal_pos[0],
                self._goal_pos[1],
                min(dist, 1.0),
                step_frac,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "agent_pos": self._agent_pos.tolist(),
            "goal_pos": self._goal_pos.tolist(),
            "step_count": self._step_count,
            "hazard_positions": [h.tolist() for h in self._hazard_positions],
        }
