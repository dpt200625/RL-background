"""Tabular policies pi(a | s).

All policies expose the same two methods:
    probs(s)       -> vector pi(. | s)
    sample(s, rng) -> an action drawn from pi(. | s)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def greedy_actions(values: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """All maximizing actions of a value row (ties are kept, not broken)."""
    return np.flatnonzero(values >= values.max() - atol)


class Policy(ABC):
    n_actions: int

    @abstractmethod
    def probs(self, s: int) -> np.ndarray:
        ...

    def sample(self, s: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probs(s)))

    def as_matrix(self, n_states: int) -> np.ndarray:
        return np.stack([self.probs(s) for s in range(n_states)])


class TabularPolicy(Policy):
    """A policy stored as a table pi[s, a]."""

    def __init__(self, table: np.ndarray):
        table = np.asarray(table, dtype=float)
        if not np.allclose(table.sum(axis=1), 1.0):
            raise ValueError("each row of a policy table must sum to 1")
        self.table = table
        self.n_states, self.n_actions = table.shape

    @classmethod
    def uniform(cls, n_states: int, n_actions: int) -> "TabularPolicy":
        return cls(np.full((n_states, n_actions), 1.0 / n_actions))

    @classmethod
    def deterministic(cls, actions: np.ndarray, n_actions: int) -> "TabularPolicy":
        table = np.zeros((len(actions), n_actions))
        table[np.arange(len(actions)), actions] = 1.0
        return cls(table)

    @classmethod
    def greedy(cls, q: np.ndarray, tie_break: str = "first") -> "TabularPolicy":
        """pi'(s) in argmax_a q(s, a). tie_break: 'first' (deterministic rule) or 'uniform' over ties."""
        table = np.zeros_like(q, dtype=float)
        for s in range(q.shape[0]):
            best = greedy_actions(q[s])
            if tie_break == "first":
                table[s, best[0]] = 1.0
            else:
                table[s, best] = 1.0 / len(best)
        return cls(table)

    def probs(self, s: int) -> np.ndarray:
        return self.table[s]

    def as_matrix(self, n_states: int | None = None) -> np.ndarray:
        return self.table


class EpsilonGreedyPolicy(Policy):
    """epsilon-greedy with respect to a *live* action-value table Q.

    With probability 1 - epsilon pick a greedy action (uniformly among ties),
    otherwise pick an action uniformly at random. Because the policy keeps a
    reference to Q, it changes as soon as the learner updates Q.
    """

    def __init__(self, q: np.ndarray, epsilon: float):
        self.q = q
        self.epsilon = epsilon
        self.n_actions = q.shape[1]

    def probs(self, s: int) -> np.ndarray:
        p = np.full(self.n_actions, self.epsilon / self.n_actions)
        best = greedy_actions(self.q[s])
        p[best] += (1.0 - self.epsilon) / len(best)
        return p

    def sample(self, s: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        best = greedy_actions(self.q[s])
        return int(best[0] if len(best) == 1 else rng.choice(best))
