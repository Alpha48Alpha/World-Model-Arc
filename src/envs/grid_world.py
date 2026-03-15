"""Grid-based simulated world environment.

A configurable 2-D grid world where the agent navigates from a start cell
to one or more goal cells while avoiding walls and optional hazards.
The environment follows the Gymnasium API so it is drop-in compatible with
standard RL tooling.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorld(gym.Env):
    """Discrete 2-D grid world with configurable goals, walls, and hazards.

    Observation:
        Flat integer array of shape (height * width,) with values in
        {0=empty, 1=wall, 2=agent, 3=goal, 4=hazard}.

    Actions:
        Discrete(4): 0=up, 1=right, 2=down, 3=left.

    Reward:
        +1.0 on reaching a goal, -1.0 on stepping into a hazard,
        -0.01 per step (time penalty), 0 otherwise.
    """

    metadata = {"render_modes": ["ansi"]}

    # Cell types
    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3
    HAZARD = 4

    _ACTION_DELTAS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        n_goals: int = 1,
        n_hazards: int = 2,
        n_walls: int = 5,
        max_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.n_goals = n_goals
        self.n_hazards = n_hazards
        self.n_walls = n_walls
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(height * width,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._rng = np.random.default_rng(seed)
        self._grid: np.ndarray = np.zeros((height, width), dtype=np.int8)
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_positions: list[tuple[int, int]] = []
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

        self._grid = np.zeros((self.height, self.width), dtype=np.int8)
        self._step_count = 0

        all_cells = [(r, c) for r in range(self.height) for c in range(self.width)]
        # Use numpy rng for reproducible shuffling
        idx = self._rng.permutation(len(all_cells))
        shuffled = [all_cells[i] for i in idx]

        # Place agent
        self._agent_pos = shuffled.pop()

        # Place goals
        self._goal_positions = []
        for _ in range(self.n_goals):
            pos = shuffled.pop()
            self._goal_positions.append(pos)
            self._grid[pos] = self.GOAL

        # Place hazards
        for _ in range(self.n_hazards):
            pos = shuffled.pop()
            self._grid[pos] = self.HAZARD

        # Place walls
        for _ in range(self.n_walls):
            pos = shuffled.pop()
            self._grid[pos] = self.WALL

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        dr, dc = self._ACTION_DELTAS[int(action)]
        nr = self._agent_pos[0] + dr
        nc = self._agent_pos[1] + dc

        # Clamp to grid bounds
        nr = int(np.clip(nr, 0, self.height - 1))
        nc = int(np.clip(nc, 0, self.width - 1))

        reward = -0.01  # time penalty
        terminated = False

        cell = self._grid[nr, nc]
        if cell == self.WALL:
            # Bounce back — stay in place
            nr, nc = self._agent_pos

        elif cell == self.GOAL:
            reward = 1.0
            terminated = True

        elif cell == self.HAZARD:
            reward = -1.0
            terminated = True

        self._agent_pos = (nr, nc)
        truncated = (not terminated) and (self._step_count >= self.max_steps)
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        symbols = {self.EMPTY: ".", self.WALL: "#", self.GOAL: "G", self.HAZARD: "X"}
        rows = []
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                if (r, c) == self._agent_pos:
                    row_str += "A"
                else:
                    row_str += symbols.get(int(self._grid[r, c]), "?")
            rows.append(row_str)
        board = "\n".join(rows)
        print(board)
        return board

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        grid = self._grid.copy().astype(np.float32)
        grid[self._agent_pos] = self.AGENT
        return grid.flatten()

    def _get_info(self) -> dict[str, Any]:
        return {
            "agent_pos": self._agent_pos,
            "goal_positions": list(self._goal_positions),
            "step_count": self._step_count,
        }
