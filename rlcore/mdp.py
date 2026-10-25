"""Finite MDP with a known model, written in Silver's notation.

A finite MDP is the tuple (S, A, P, R, gamma):

    P[s, a, s'] = Pr(S_{t+1} = s' | S_t = s, A_t = a)        (P^a_{ss'})
    R[s, a]     = E[R_{t+1} | S_t = s, A_t = a]               (R^a_s)

The model is stored as a list of outcomes per (s, a), the same format as
gymnasium's toy-text environments:

    outcomes[s][a] = [(prob, next_state, reward), ...]

so a sampled transition keeps its exact reward, while the dense arrays P and R
are derived once for dynamic programming. Terminal states are absorbing and
have value zero, v(S_T) = 0.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np

Outcome = tuple[float, int, float]


@dataclass(frozen=True)
class Step:
    """What the environment returns after one action.

    terminated: the task really ended (S_{t+1} is terminal, its value is zero).
    truncated:  the run was cut by a time limit; the task itself did not end.
    """

    next_state: int
    reward: float
    terminated: bool
    truncated: bool = False

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class TabularMDP:
    """Finite MDP whose model is known (used for DP) and that can also be sampled (used for MC/TD)."""

    def __init__(
        self,
        outcomes: Sequence[Sequence[Sequence[Outcome]]],
        terminal: Sequence[bool],
        start_dist: Sequence[float],
        gamma: float = 1.0,
        state_names: Sequence[str] | None = None,
        action_names: Sequence[str] | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        self.n_states = len(outcomes)
        self.n_actions = len(outcomes[0])
        self.terminal = np.asarray(terminal, dtype=bool)
        self.start_dist = np.asarray(start_dist, dtype=float)
        self.gamma = gamma
        self.state_names = list(state_names) if state_names is not None else [str(s) for s in range(self.n_states)]
        self.action_names = list(action_names) if action_names is not None else [str(a) for a in range(self.n_actions)]
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.outcomes = [[list(outcomes[s][a]) for a in range(self.n_actions)] for s in range(self.n_states)]
        for s in np.flatnonzero(self.terminal):  # absorbing, zero reward
            self.outcomes[s] = [[(1.0, int(s), 0.0)] for _ in range(self.n_actions)]
        self._build_arrays()

        self.state: int | None = None
        self.t = 0

    # ------------------------------------------------------------------ model
    def _build_arrays(self) -> None:
        S, A = self.n_states, self.n_actions
        self.P = np.zeros((S, A, S))
        self.R = np.zeros((S, A))
        self._cum_probs = [[None] * A for _ in range(S)]
        for s in range(S):
            for a in range(A):
                outs = self.outcomes[s][a]
                probs = np.array([p for p, _, _ in outs])
                if not np.isclose(probs.sum(), 1.0):
                    raise ValueError(f"outcome probabilities of (s={s}, a={a}) sum to {probs.sum()}")
                for p, ns, r in outs:
                    self.P[s, a, ns] += p
                    self.R[s, a] += p * r
                self._cum_probs[s][a] = np.cumsum(probs).tolist()

    @classmethod
    def from_transitions(
        cls,
        states: Sequence[Hashable],
        actions: Sequence[Hashable],
        transitions: Mapping[tuple[Hashable, Hashable | None], Sequence[tuple[float, Hashable, float]]],
        terminal_states: Sequence[Hashable],
        start: Hashable,
        gamma: float = 1.0,
        **kwargs,
    ) -> "TabularMDP":
        """Build a small MDP by names, e.g. the worked examples of the lecture.

        transitions[(s, a)] = [(prob, s_next, reward), ...]; the key (s, None)
        means "every action in s behaves like this".
        """
        s_idx = {s: i for i, s in enumerate(states)}
        a_idx = {a: i for i, a in enumerate(actions)}
        outcomes = [[[] for _ in actions] for _ in states]
        for (s, a), outs in transitions.items():
            targets = actions if a is None else [a]
            for b in targets:
                outcomes[s_idx[s]][a_idx[b]] = [(p, s_idx[ns], r) for p, ns, r in outs]
        terminal = [s in terminal_states for s in states]
        for s, row in zip(states, outcomes):
            if s not in terminal_states and any(len(outs) == 0 for outs in row):
                raise ValueError(f"state {s!r} is missing transitions for some actions")
        start_dist = np.zeros(len(states))
        start_dist[s_idx[start]] = 1.0
        return cls(outcomes, terminal, start_dist, gamma,
                   state_names=[str(s) for s in states], action_names=[str(a) for a in actions], **kwargs)

    def state_index(self, name: str) -> int:
        return self.state_names.index(name)

    def action_index(self, name: str) -> int:
        return self.action_names.index(name)

    def lookahead(self, v: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """One-step lookahead q[s, a] = R^a_s + gamma * sum_s' P^a_{ss'} v(s')  (Eq. 6, Lesson 2).

        Terminal states contribute value zero, so a branch into a terminal only
        contributes its immediate reward.
        """
        gamma = self.gamma if gamma is None else gamma
        v = np.where(self.terminal, 0.0, v)
        return self.R + gamma * self.P @ v

    def policy_model(self, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapse the MDP under policy pi[s, a] into a Markov reward process (P_pi, R_pi)."""
        P_pi = np.einsum("sa,sat->st", pi, self.P)
        R_pi = (pi * self.R).sum(axis=1)
        return P_pi, R_pi

    def evaluate_policy_exact(self, pi: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """Exact v_pi by solving the linear Bellman expectation system (I - gamma P_pi) v = R_pi.

        Used as ground truth to check the iterative and sample-based estimators.
        With gamma = 1 the policy must reach a terminal state with probability one.
        """
        gamma = self.gamma if gamma is None else gamma
        P_pi, R_pi = self.policy_model(pi)
        live = ~self.terminal
        A = np.eye(live.sum()) - gamma * P_pi[np.ix_(live, live)]
        v = np.zeros(self.n_states)
        v[live] = np.linalg.solve(A, R_pi[live])
        return v

    # ------------------------------------------------------------ interaction
    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> int:
        self.state = int(self.rng.choice(self.n_states, p=self.start_dist))
        self.t = 0
        return self.state

    def step(self, action: int) -> Step:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        outs = self.outcomes[self.state][action]
        if len(outs) == 1:
            _, ns, r = outs[0]
        else:
            i = bisect_right(self._cum_probs[self.state][action], self.rng.random())
            _, ns, r = outs[min(i, len(outs) - 1)]
        self.state = ns
        self.t += 1
        terminated = bool(self.terminal[ns])
        truncated = (not terminated) and self.max_steps is not None and self.t >= self.max_steps
        return Step(ns, r, terminated, truncated)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_states={self.n_states}, n_actions={self.n_actions}, gamma={self.gamma})"
