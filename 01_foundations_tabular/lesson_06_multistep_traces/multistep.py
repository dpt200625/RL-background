"""Lesson 6 -- multi-step returns and eligibility traces (tabular, on-policy prediction).

    Eq. (18)  G_t^(n) = sum_{k=0}^{n-1} gamma^k R_{t+k+1} + gamma^n V(S_{t+n})
    Eq. (19)  E_t(s) = gamma lambda E_{t-1}(s) + 1{S_t = s},   E_{-1}(s) = 0
    Eq. (20)  V(s) <- V(s) + alpha delta_t E_t(s)        for every s

Forward view: the lambda-return averages the n-step returns with weights
(1 - lambda) lambda^{n-1}. Backward view: one TD error is sent to all states
that still carry a trace. They coincide exactly in the offline setting (V
frozen during the episode, updates applied at its end); online accumulating
TD(lambda) is only close to it.

Run:  python multistep.py
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rlcore import Trajectory


def n_step_return(rewards: Sequence[float], states: Sequence[int], t: int, n: int, gamma: float,
                  V: np.ndarray, terminated: bool = True) -> float:
    """Eq. (18) for time t of an episode with T = len(rewards) transitions.

    If t + n reaches the terminal, the sum stops there and there is no bootstrap
    (the terminal has value zero). Otherwise append gamma^n V(S_{t+n}).
    """
    T = len(rewards)
    end = min(t + n, T)
    G = sum(gamma ** (k - t) * rewards[k] for k in range(t, end))
    if t + n < T or (t + n == T and not terminated):
        G += gamma ** (end - t) * V[states[end]]
    return float(G)


def lambda_return(rewards: Sequence[float], states: Sequence[int], t: int, gamma: float, lam: float,
                  V: np.ndarray) -> float:
    """Forward view: G_t^lambda = (1-lambda) sum_{n=1}^{T-t-1} lambda^{n-1} G_t^(n) + lambda^{T-t-1} G_t."""
    T = len(rewards)
    G = sum((1 - lam) * lam ** (n - 1) * n_step_return(rewards, states, t, n, gamma, V) for n in range(1, T - t))
    return float(G + lam ** (T - t - 1) * n_step_return(rewards, states, t, T - t, gamma, V))


class AccumulatingTrace:
    """Eq. (19): the trace of a state decays by gamma*lambda each step and gains 1 on every visit.

    Revisits accumulate, so a trace is not a probability and can exceed 1.
    """

    def __init__(self, n_states: int, gamma: float, lam: float):
        self.decay = gamma * lam
        self.e = np.zeros(n_states)

    def reset(self) -> None:
        self.e[:] = 0.0  # E_{-1}(s) = 0 at the start of every episode

    def visit(self, s: int) -> np.ndarray:
        self.e *= self.decay
        self.e[s] += 1.0
        return self.e


class NStepTDPrediction:
    """n-step TD (Sutton-Barto 7.1): update V(S_tau) with G_tau^(n) as soon as n rewards are known."""

    def __init__(self, n_states: int, n: int, alpha: float, gamma: float = 1.0, init_value: float = 0.0):
        self.V = np.full(n_states, init_value, dtype=float)
        self.n, self.alpha, self.gamma = n, alpha, gamma

    def learn_episode(self, traj: Trajectory) -> None:
        states, rewards = traj.states, traj.rewards
        T = len(rewards)
        for tau in range(T):  # tau = t - n + 1 runs over every visited time step, in order
            G = n_step_return(rewards, states, tau, self.n, self.gamma, self.V, traj.terminated)
            self.V[states[tau]] += self.alpha * (G - self.V[states[tau]])


class TDLambdaPrediction:
    """Online TD(lambda), backward view with accumulating traces (Eqs. 19-20)."""

    def __init__(self, n_states: int, lam: float, alpha: float, gamma: float = 1.0, init_value: float = 0.0):
        self.V = np.full(n_states, init_value, dtype=float)
        self.lam, self.alpha, self.gamma = lam, alpha, gamma
        self.trace = AccumulatingTrace(n_states, gamma, lam)

    def learn_episode(self, traj: Trajectory) -> None:
        self.trace.reset()
        for s, _, r, s_next, terminal in traj.transitions():
            delta = r + (0.0 if terminal else self.gamma * self.V[s_next]) - self.V[s]
            e = self.trace.visit(s)
            self.V += self.alpha * delta * e


def offline_lambda_return_update(V: np.ndarray, traj: Trajectory, alpha: float, gamma: float, lam: float) -> np.ndarray:
    """Forward view, offline: V is frozen while all lambda-return targets are formed."""
    states, rewards = traj.states, traj.rewards
    dV = np.zeros_like(V)
    for t in range(len(rewards)):
        dV[states[t]] += alpha * (lambda_return(rewards, states, t, gamma, lam, V) - V[states[t]])
    return V + dV


def offline_td_lambda_update(V: np.ndarray, traj: Trajectory, alpha: float, gamma: float, lam: float) -> np.ndarray:
    """Backward view, offline: TD errors use the frozen V, the sum of alpha delta_t E_t is applied at the end."""
    trace = AccumulatingTrace(len(V), gamma, lam)
    dV = np.zeros_like(V)
    for s, _, r, s_next, terminal in traj.transitions():
        delta = r + (0.0 if terminal else gamma * V[s_next]) - V[s]
        dV += alpha * delta * trace.visit(s)
    return V + dV


if __name__ == "__main__":
    V = np.array([0.0, 0.0, 3.0])
    print("G^(2) for rewards 1, 2, V(S_{t+2}) = 3:", round(n_step_return([1, 2, 0], [0, 1, 2, 2], 0, 2, 0.9, V), 4))
    print("ends after the two rewards:", round(n_step_return([1, 2], [0, 1, 2], 0, 2, 0.9, V), 4))
    tr = AccumulatingTrace(3, gamma=1.0, lam=0.8)
    print("trace of a state visited once:", [round(float(tr.visit(s)[0]), 4) for s in (0, 1, 2)])
    print("revisit when the trace is 0.64:", round(float(tr.visit(0)[0]), 4))
