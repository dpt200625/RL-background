"""Lesson 1 -- observation vs Markov state (Exercise 2).

An object moves on a ring of cells with a hidden velocity of -1 or +1.
The true state is (position, velocity); the camera only sees the position.
From one frame the same observation can be followed by different next
observations, so the single frame is not a Markov state. Two stacked frames
reveal the velocity and make the next observation predictable again.

Run:  python observability.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class MovingObject:
    """Hidden state (x, v); observation = x only. The velocity flips with probability `flip_prob`."""

    def __init__(self, n_cells: int = 6, flip_prob: float = 0.1, seed: int | None = None):
        self.n_cells, self.flip_prob = n_cells, flip_prob
        self.rng = np.random.default_rng(seed)
        self.x, self.v = 0, 1

    def reset(self) -> int:
        self.x = int(self.rng.integers(self.n_cells))
        self.v = int(self.rng.choice([-1, 1]))
        return self.observe()

    def observe(self) -> int:
        return self.x

    def step(self) -> int:
        if self.rng.random() < self.flip_prob:
            self.v = -self.v
        self.x = (self.x + self.v) % self.n_cells
        return self.observe()


def next_observation_table(env: MovingObject, history_len: int, n_steps: int = 5_000) -> dict:
    """Empirical distribution of the next observation given the last `history_len` observations."""
    counts: dict[tuple, defaultdict] = defaultdict(lambda: defaultdict(int))
    history = [env.reset()]
    for _ in range(n_steps):
        obs = env.step()
        if len(history) >= history_len:
            counts[tuple(history[-history_len:])][obs] += 1
        history.append(obs)
    return {k: {o: c / sum(v.values()) for o, c in v.items()} for k, v in counts.items()}


def predictability(table: dict) -> float:
    """Average probability of the most likely next observation (1.0 = fully predictable)."""
    return float(np.mean([max(dist.values()) for dist in table.values()]))


if __name__ == "__main__":
    for h in (1, 2):
        table = next_observation_table(MovingObject(seed=0), history_len=h)
        print(f"history of {h} frame(s): next observation predictability = {predictability(table):.2f}")
