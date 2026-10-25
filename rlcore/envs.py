"""Small tabular environments used across Chapter I.

* GridWorld    -- grid navigation with walls, goal, cliff and optional slip.
* cliff_walking -- Sutton-Barto Example 6.6 (Sarsa vs Q-learning).
* RandomWalk   -- Sutton-Barto Example 6.2 / 7.1 (MC vs TD, n-step, TD(lambda)).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .mdp import TabularMDP

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = "^>v<"


class GridWorld(TabularMDP):
    """Grid navigation with a known map.

    Layout characters:
        S start, G goal (terminal), # wall, C cliff, . free cell.

    Every move costs `step_reward`. Bumping into a wall or the border leaves the
    agent in place. Entering a cliff cell gives `cliff_reward` and sends the
    agent back to S without ending the episode. With `slip > 0` the agent moves
    in the intended direction with probability 1 - slip and to each
    perpendicular direction with probability slip / 2.
    """

    def __init__(
        self,
        layout: Sequence[str],
        slip: float = 0.0,
        step_reward: float = -1.0,
        goal_reward: float | None = None,
        cliff_reward: float = -100.0,
        gamma: float = 1.0,
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        self.layout = [list(row) for row in layout]
        self.n_rows, self.n_cols = len(layout), len(layout[0])
        self.slip = slip
        self.step_reward = step_reward
        self.goal_reward = step_reward if goal_reward is None else goal_reward
        self.cliff_reward = cliff_reward

        cells = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]
        self.start_state = self._find("S")
        terminal = [self.layout[r][c] in "G#" for r, c in cells]  # walls are never visited
        outcomes = [[self._outcomes(s, a) for a in range(4)] for s in range(len(cells))]
        start = np.zeros(len(cells))
        start[self.start_state] = 1.0
        names = [f"({r},{c})" for r, c in cells]
        super().__init__(outcomes, terminal, start, gamma, names, ["up", "right", "down", "left"], max_steps, seed)

    def to_state(self, r: int, c: int) -> int:
        return r * self.n_cols + c

    def to_pos(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def cell(self, s: int) -> str:
        r, c = self.to_pos(s)
        return self.layout[r][c]

    def _find(self, ch: str) -> int:
        for r, row in enumerate(self.layout):
            if ch in row:
                return self.to_state(r, row.index(ch))
        raise ValueError(f"layout has no {ch!r}")

    def _move(self, s: int, a: int) -> int:
        r, c = self.to_pos(s)
        nr, nc = r + MOVES[a][0], c + MOVES[a][1]
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or self.layout[nr][nc] == "#":
            return s
        return self.to_state(nr, nc)

    def _outcomes(self, s: int, a: int) -> list[tuple[float, int, float]]:
        if self.cell(s) in "G#":
            return [(1.0, s, 0.0)]
        moves = {a: 1.0 - self.slip}
        for side in ((a + 1) % 4, (a + 3) % 4):
            moves[side] = moves.get(side, 0.0) + self.slip / 2
        outs = []
        for b, p in moves.items():
            if p <= 0:
                continue
            ns = self._move(s, b)
            if self.cell(ns) == "C":
                outs.append((p, self.start_state, self.cliff_reward))
            elif self.cell(ns) == "G":
                outs.append((p, ns, self.goal_reward))
            else:
                outs.append((p, ns, self.step_reward))
        return outs

    # ------------------------------------------------------------- rendering
    def value_grid(self, v: np.ndarray) -> np.ndarray:
        """Values as an (n_rows, n_cols) array, NaN on walls (for heatmaps)."""
        grid = np.asarray(v, dtype=float).reshape(self.n_rows, self.n_cols).copy()
        for s in range(self.n_states):
            if self.cell(s) == "#":
                grid[self.to_pos(s)] = np.nan
        return grid

    def render_policy(self, policy: np.ndarray) -> str:
        """ASCII arrows; `policy` is (n_states,) actions or an (n_states, n_actions) table."""
        policy = np.asarray(policy)
        actions = policy.argmax(axis=1) if policy.ndim == 2 else policy
        rows = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                s = self.to_state(r, c)
                ch = self.cell(s)
                row += ch if ch in "G#C" else ARROWS[actions[s]]
            rows.append(row)
        return "\n".join(rows)

    def render_values(self, v: np.ndarray, fmt: str = "{:7.2f}") -> str:
        rows = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                s = self.to_state(r, c)
                cells.append("   ##  " if self.cell(s) == "#" else fmt.format(v[s]))
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def greedy_path(self, policy: np.ndarray, max_len: int = 100) -> list[int]:
        """States visited by following the most likely action from S (ignores slip)."""
        policy = np.asarray(policy)
        actions = policy.argmax(axis=1) if policy.ndim == 2 else policy
        path, s = [self.start_state], self.start_state
        for _ in range(max_len):
            if self.terminal[s]:
                break
            s = max(self.outcomes[s][int(actions[s])], key=lambda o: o[0])[1]
            path.append(s)
        return path


def cliff_walking(**kwargs) -> GridWorld:
    """Sutton-Barto Example 6.6: 4 x 12 grid, cliff along the bottom edge."""
    layout = ["............",
              "............",
              "............",
              "SCCCCCCCCCCG"]
    return GridWorld(layout, **kwargs)


class RandomWalk(TabularMDP):
    """Sutton-Barto random walk: a chain of `n` states between two terminals.

    Actions: 0 = left, 1 = right. The classic "random walk" is the uniform
    policy on this chain, starting in the middle state. Reaching the right end
    pays `right_reward`, the left end pays `left_reward`, every other step 0.

    States are numbered 0 .. n+1; 0 and n+1 are the terminals.
    n = 5,  left_reward = 0  -> Example 6.2, true values 1/6, ..., 5/6.
    n = 19, left_reward = -1 -> Example 7.1, true values -0.9, ..., 0.9.
    """

    LEFT, RIGHT = 0, 1

    def __init__(self, n: int = 5, left_reward: float = 0.0, right_reward: float = 1.0,
                 gamma: float = 1.0, seed: int | None = None):
        self.n = n
        S = n + 2
        outcomes = []
        for s in range(S):
            row = []
            for a in (self.LEFT, self.RIGHT):
                ns = s - 1 if a == self.LEFT else s + 1
                ns = min(max(ns, 0), S - 1)
                r = left_reward if ns == 0 else right_reward if ns == S - 1 else 0.0
                row.append([(1.0, ns, r)])
            outcomes.append(row)
        terminal = [s in (0, S - 1) for s in range(S)]
        start = np.zeros(S)
        start[(S - 1) // 2] = 1.0
        names = ["L"] + [chr(ord("A") + i) if n <= 26 else str(i + 1) for i in range(n)] + ["R"]
        super().__init__(outcomes, terminal, start, gamma, names, ["left", "right"], seed=seed)

    def uniform_policy(self) -> np.ndarray:
        return np.full((self.n_states, 2), 0.5)

    def true_values(self) -> np.ndarray:
        """v of the uniform random-walk policy (non-terminal states only)."""
        return self.evaluate_policy_exact(self.uniform_policy())[1:-1]
