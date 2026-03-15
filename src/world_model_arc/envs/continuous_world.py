"""Continuous-state world environment.

A 2-D continuous navigation task where the agent controls acceleration
in the x and y directions.  Features:
- Configurable workspace bounds
- Multiple goal regions (circles)
- Hazard regions (circles) that end the episode with negative reward
- Bounded velocity and position
- Dense distance-based shaping reward
- Flat observation: (x, y, vx, vy, dx_to_nearest_goal, dy_to_nearest_goal)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from world_model_arc.envs.base import BaseEnv, StepResult

_N_ACTIONS = 5  # no-op, accel +x, accel -x, accel +y, accel -y

_ACCEL = 0.3
_MAX_VEL = 1.0
_DT = 0.1
_GOAL_RADIUS = 0.15
_HAZARD_RADIUS = 0.12

_REWARD_GOAL = 1.0
_REWARD_HAZARD = -1.0
_REWARD_STEP = -0.005


class ContinuousWorld(BaseEnv):
    """Continuous 2-D navigation environment with discrete action set.

    The state lives in the unit square ``[0, 1] × [0, 1]``.
    The agent is a point mass with first-order velocity dynamics.

    Args:
        goals: List of goal centre positions ``(x, y)``.
        hazards: List of hazard centre positions ``(x, y)``.
        max_steps: Episode time limit.
        dense_reward: If *True*, add a shaping term proportional to
            the change in distance to the nearest goal.
    """

    def __init__(
        self,
        goals: Optional[list[tuple[float, float]]] = None,
        hazards: Optional[list[tuple[float, float]]] = None,
        max_steps: int = 300,
        dense_reward: bool = True,
    ) -> None:
        self.goals: list[tuple[float, float]] = goals or [(0.85, 0.85)]
        self.hazards: list[tuple[float, float]] = hazards or [(0.5, 0.5)]
        self.max_steps = max_steps
        self.dense_reward = dense_reward

        self._rng = np.random.default_rng()
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_goal_dist = 0.0

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple[int, ...]:
        # (x, y, vx, vy, dx_goal, dy_goal, steps_remaining)
        return (7,)

    @property
    def action_size(self) -> int:
        return _N_ACTIONS

    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Start away from goals / hazards
        while True:
            self._pos = self._rng.random(2).astype(np.float32) * 0.4 + 0.05
            if not self._in_any_hazard(self._pos) and not self._in_any_goal(self._pos):
                break
        self._vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_goal_dist = self._dist_to_nearest_goal(self._pos)
        return self._observe()

    def step(self, action: int) -> StepResult:
        assert 0 <= action < _N_ACTIONS

        ax, ay = 0.0, 0.0
        if action == 1:
            ax = _ACCEL
        elif action == 2:
            ax = -_ACCEL
        elif action == 3:
            ay = _ACCEL
        elif action == 4:
            ay = -_ACCEL

        self._vel[0] = float(np.clip(self._vel[0] + ax * _DT, -_MAX_VEL, _MAX_VEL))
        self._vel[1] = float(np.clip(self._vel[1] + ay * _DT, -_MAX_VEL, _MAX_VEL))
        self._pos = np.clip(self._pos + self._vel * _DT, 0.0, 1.0).astype(np.float32)

        self._step_count += 1

        reward = _REWARD_STEP
        done = False
        info: dict = {}

        if self._in_any_goal(self._pos):
            reward += _REWARD_GOAL
            done = True
            info["goal_reached"] = True
        elif self._in_any_hazard(self._pos):
            reward += _REWARD_HAZARD
            done = True
            info["hazard_hit"] = True
        elif self.dense_reward:
            cur_dist = self._dist_to_nearest_goal(self._pos)
            reward += 0.1 * (self._prev_goal_dist - cur_dist)
            self._prev_goal_dist = cur_dist

        truncated = (not done) and (self._step_count >= self.max_steps)
        return StepResult(
            observation=self._observe(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def render(self) -> np.ndarray:
        """Return a simple 128×128 RGB image of the world state."""
        size = 128
        img = np.ones((size, size, 3), dtype=np.uint8) * 240

        def _to_px(xy: np.ndarray) -> tuple[int, int]:
            x = int(np.clip(xy[0] * (size - 1), 0, size - 1))
            y = int(np.clip((1 - xy[1]) * (size - 1), 0, size - 1))
            return y, x  # row, col

        # Draw goals (green)
        for gx, gy in self.goals:
            r, c = _to_px(np.array([gx, gy]))
            rr = int(_GOAL_RADIUS * size)
            for dr in range(-rr, rr + 1):
                for dc in range(-rr, rr + 1):
                    if dr * dr + dc * dc <= rr * rr:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            img[nr, nc] = (50, 205, 50)

        # Draw hazards (red)
        for hx, hy in self.hazards:
            r, c = _to_px(np.array([hx, hy]))
            rr = int(_HAZARD_RADIUS * size)
            for dr in range(-rr, rr + 1):
                for dc in range(-rr, rr + 1):
                    if dr * dr + dc * dc <= rr * rr:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            img[nr, nc] = (220, 20, 60)

        # Draw agent (blue)
        r, c = _to_px(self._pos)
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    img[nr, nc] = (30, 144, 255)

        return img

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist_to_nearest_goal(self, pos: np.ndarray) -> float:
        return float(min(np.linalg.norm(pos - np.array(g)) for g in self.goals))

    def _in_any_goal(self, pos: np.ndarray) -> bool:
        return any(np.linalg.norm(pos - np.array(g)) <= _GOAL_RADIUS for g in self.goals)

    def _in_any_hazard(self, pos: np.ndarray) -> bool:
        return any(np.linalg.norm(pos - np.array(h)) <= _HAZARD_RADIUS for h in self.hazards)

    def _observe(self) -> np.ndarray:
        gx, gy = self.goals[0]
        dx = gx - float(self._pos[0])
        dy = gy - float(self._pos[1])
        steps_rem = max(0, self.max_steps - self._step_count) / self.max_steps
        return np.array(
            [
                float(self._pos[0]),
                float(self._pos[1]),
                float(self._vel[0]),
                float(self._vel[1]),
                dx,
                dy,
                steps_rem,
            ],
            dtype=np.float32,
        )
