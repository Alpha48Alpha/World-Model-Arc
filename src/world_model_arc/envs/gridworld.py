"""Grid-world environment.

A rich, configurable discrete 2-D grid world with:
- Multi-goal support (multiple reward tiles)
- Hazard tiles (negative reward, episode ends)
- Wall tiles (impassable)
- Step-count limit
- Optional stochastic transitions (slip probability)
- Flat and one-hot observation modes
- RGB rendering
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from world_model_arc.envs.base import BaseEnv, StepResult


class Tile(IntEnum):
    """Tile types in the grid."""

    EMPTY = 0
    WALL = 1
    GOAL = 2
    HAZARD = 3
    AGENT = 4


# Reward constants
_REWARD_GOAL = 1.0
_REWARD_HAZARD = -1.0
_REWARD_STEP = -0.01

# ANSI colours for render(mode="ansi")
_TILE_CHAR = {
    Tile.EMPTY: ".",
    Tile.WALL: "#",
    Tile.GOAL: "G",
    Tile.HAZARD: "X",
    Tile.AGENT: "A",
}

# RGB colours for render(mode="rgb_array")
_TILE_RGB = {
    Tile.EMPTY: (255, 255, 255),
    Tile.WALL: (40, 40, 40),
    Tile.GOAL: (50, 205, 50),
    Tile.HAZARD: (220, 20, 60),
    Tile.AGENT: (30, 144, 255),
}

# Actions: up, right, down, left
_ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N_ACTIONS = len(_ACTIONS)


class GridWorld(BaseEnv):
    """Discrete grid-world environment.

    Args:
        height: Number of rows.
        width: Number of columns.
        goals: List of ``(row, col)`` goal positions.  Defaults to bottom-right.
        hazards: List of ``(row, col)`` hazard positions.
        walls: List of ``(row, col)`` wall positions.
        start: Fixed start position ``(row, col)``.  If *None*, random empty tile.
        max_steps: Maximum steps per episode.
        slip_prob: Probability of taking a random action instead of the chosen one.
        obs_mode: ``"flat"`` (normalised coords) or ``"onehot"`` (full grid one-hot).
    """

    def __init__(
        self,
        height: int = 8,
        width: int = 8,
        goals: Optional[list[tuple[int, int]]] = None,
        hazards: Optional[list[tuple[int, int]]] = None,
        walls: Optional[list[tuple[int, int]]] = None,
        start: Optional[tuple[int, int]] = None,
        max_steps: int = 200,
        slip_prob: float = 0.0,
        obs_mode: str = "flat",
    ) -> None:
        self.height = height
        self.width = width
        self.goals: list[tuple[int, int]] = goals if goals is not None else [(height - 1, width - 1)]
        self.hazards: list[tuple[int, int]] = hazards or []
        self.walls: list[tuple[int, int]] = walls or []
        self.start = start
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self.obs_mode = obs_mode

        assert obs_mode in {"flat", "onehot"}, f"Unknown obs_mode: {obs_mode}"

        self._rng = np.random.default_rng()
        self._grid = self._build_base_grid()
        self._agent_pos: tuple[int, int] = (0, 0)
        self._step_count: int = 0
        self._visited_goals: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def observation_shape(self) -> tuple[int, ...]:
        if self.obs_mode == "flat":
            # (row, col, steps_remaining) – all normalised to [0, 1]
            return (3,)
        # one-hot over tile types for every cell + agent position channel
        return (len(Tile), self.height, self.width)

    @property
    def action_size(self) -> int:
        return _N_ACTIONS

    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._grid = self._build_base_grid()
        self._step_count = 0
        self._visited_goals = set()
        if self.start is not None:
            self._agent_pos = self.start
        else:
            self._agent_pos = self._random_empty_tile()
        return self._observe()

    def step(self, action: int) -> StepResult:
        assert 0 <= action < _N_ACTIONS, f"Invalid action: {action}"

        # Stochastic slip
        if self._rng.random() < self.slip_prob:
            action = int(self._rng.integers(0, _N_ACTIONS))

        dr, dc = _ACTIONS[action]
        nr = self._agent_pos[0] + dr
        nc = self._agent_pos[1] + dc

        # Clamp to grid and respect walls
        nr = int(np.clip(nr, 0, self.height - 1))
        nc = int(np.clip(nc, 0, self.width - 1))
        if self._grid[nr, nc] == Tile.WALL:
            nr, nc = self._agent_pos  # bounce back

        self._agent_pos = (nr, nc)
        self._step_count += 1

        reward = _REWARD_STEP
        done = False
        info: dict = {}

        tile = self._grid[nr, nc]
        if tile == Tile.GOAL:
            reward += _REWARD_GOAL
            self._visited_goals.add((nr, nc))
            info["goal_reached"] = True
            if set(self.goals) == self._visited_goals:
                done = True  # all goals collected
        elif tile == Tile.HAZARD:
            reward += _REWARD_HAZARD
            done = True
            info["hazard_hit"] = True

        truncated = (not done) and (self._step_count >= self.max_steps)
        return StepResult(
            observation=self._observe(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def render(self) -> np.ndarray:
        """Return an RGB image (H, W, 3) of the current grid state."""
        cell = max(16, 512 // max(self.height, self.width))
        img = np.zeros((self.height * cell, self.width * cell, 3), dtype=np.uint8)
        for r in range(self.height):
            for c in range(self.width):
                tile = Tile(int(self._grid[r, c]))
                if (r, c) == self._agent_pos:
                    tile = Tile.AGENT
                colour = _TILE_RGB[tile]
                rr, cc = r * cell, c * cell
                img[rr : rr + cell, cc : cc + cell] = colour
        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_grid(self) -> np.ndarray:
        grid = np.full((self.height, self.width), Tile.EMPTY, dtype=np.int32)
        for r, c in self.walls:
            grid[r, c] = Tile.WALL
        for r, c in self.hazards:
            grid[r, c] = Tile.HAZARD
        for r, c in self.goals:
            grid[r, c] = Tile.GOAL
        return grid

    def _random_empty_tile(self) -> tuple[int, int]:
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self._grid[r, c] == Tile.EMPTY
            and (r, c) not in self.goals
        ]
        idx = int(self._rng.integers(0, len(empty)))
        return empty[idx]

    def _observe(self) -> np.ndarray:
        if self.obs_mode == "flat":
            r, c = self._agent_pos
            steps_rem = max(0, self.max_steps - self._step_count) / self.max_steps
            return np.array(
                [r / max(1, self.height - 1), c / max(1, self.width - 1), steps_rem],
                dtype=np.float32,
            )
        # One-hot tensor: (n_tile_types, H, W)
        obs = np.zeros((len(Tile), self.height, self.width), dtype=np.float32)
        for r in range(self.height):
            for c in range(self.width):
                obs[int(self._grid[r, c]), r, c] = 1.0
        ar, ac = self._agent_pos
        obs[Tile.AGENT, ar, ac] = 1.0
        return obs
