"""Lesson 4 -- Monte Carlo and temporal-difference prediction of V ~= v_pi (policy fixed).

    Eq. (12)  V(S_t) <- V(S_t) + alpha [G_t - V(S_t)]              Monte Carlo
    Eq. (13)  delta_t = R_{t+1} + gamma V(S_{t+1}) - V(S_t)        TD error
    Eq. (14)  V(S_t) <- V(S_t) + alpha delta_t                     TD(0)

Both learn from sampled experience only: the transition model is never read.

Run:  python prediction.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rlcore import Trajectory


class ValuePredictor(ABC):
    """A table V(s) that is improved from trajectories of a fixed policy."""

    def __init__(self, n_states: int, gamma: float = 1.0, init_value: float = 0.0,
                 terminal: np.ndarray | None = None):
        self.V = np.full(n_states, init_value, dtype=float)
        self.gamma = gamma
        self.terminal = np.zeros(n_states, bool) if terminal is None else np.asarray(terminal, bool)
        self.V[self.terminal] = 0.0  # v(S_T) = 0

    @abstractmethod
    def learn_episode(self, traj: Trajectory) -> None:
        ...


class MonteCarloPrediction(ValuePredictor):
    """Eq. (12) with the full return G_t as target: wait for the episode to end.

    alpha=None uses alpha = 1 / N(S_t), i.e. the sample mean of the returns seen.
    first_visit=True updates only the first occurrence of a state in an episode;
    False updates every occurrence (every-visit MC).
    """

    def __init__(self, n_states: int, gamma: float = 1.0, alpha: float | None = None,
                 first_visit: bool = True, **kwargs):
        super().__init__(n_states, gamma, **kwargs)
        self.alpha, self.first_visit = alpha, first_visit
        self.N = np.zeros(n_states, dtype=int)

    def target(self, rewards: list[float], t: int) -> float:
        """G_t, the realized return from time t (Lesson 1, Eq. 1)."""
        return float(sum(self.gamma**k * r for k, r in enumerate(rewards[t:])))

    def update(self, s: int, G: float) -> float:
        """Move V(s) toward the target G; returns the prediction error G - V(s)."""
        self.N[s] += 1
        alpha = 1.0 / self.N[s] if self.alpha is None else self.alpha
        error = G - self.V[s]
        self.V[s] += alpha * error
        return error

    def learn_episode(self, traj: Trajectory) -> None:
        if not traj.terminated:
            raise ValueError("Monte Carlo needs a complete episode that reached a terminal state")
        states, rewards = traj.states[:-1], traj.rewards
        first = {}
        for t, s in enumerate(states):
            first.setdefault(s, t)
        G = 0.0
        for t in reversed(range(len(states))):  # backward pass: G_t = R_{t+1} + gamma G_{t+1}
            G = rewards[t] + self.gamma * G
            if not self.first_visit or first[states[t]] == t:
                self.update(states[t], G)


class TD0Prediction(ValuePredictor):
    """Eqs. (13)-(14): bootstrap from the current estimate of the next state after every transition."""

    def __init__(self, n_states: int, gamma: float = 1.0, alpha: float = 0.1, **kwargs):
        super().__init__(n_states, gamma, **kwargs)
        self.alpha = alpha

    def target(self, r: float, s_next: int, terminal: bool) -> float:
        """R_{t+1} + gamma V(S_{t+1}), with V(S_{t+1}) = 0 when S_{t+1} is terminal."""
        return r + (0.0 if terminal else self.gamma * self.V[s_next])

    def td_error(self, s: int, r: float, s_next: int, terminal: bool) -> float:
        return self.target(r, s_next, terminal) - self.V[s]

    def update(self, s: int, r: float, s_next: int, terminal: bool) -> float:
        """One update from a single transition, possible before the episode ends; returns delta_t."""
        delta = self.td_error(s, r, s_next, terminal)
        self.V[s] += self.alpha * delta
        return delta

    def learn_episode(self, traj: Trajectory) -> None:
        for s, _, r, s_next, terminal in traj.transitions():
            self.update(s, r, s_next, terminal)


if __name__ == "__main__":
    # Section 4: V(s) = 1, V(u) = 2, s -> u with reward 0, gamma = 0.9, alpha = 0.5
    td = TD0Prediction(n_states=2, gamma=0.9, alpha=0.5)
    td.V[:] = [1.0, 2.0]
    print("TD target", td.target(0.0, 1, False), "-> new V(s) =", td.V[0] + 0.5 * td.td_error(0, 0.0, 1, False))
    mc = MonteCarloPrediction(n_states=2, gamma=0.9, alpha=0.5)
    mc.V[:] = [1.0, 2.0]
    G = mc.target([0.0, 4.0], 0)
    mc.update(0, G)
    print("MC return", round(G, 4), "-> new V(s) =", round(mc.V[0], 4))
