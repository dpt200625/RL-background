"""Lesson 1 -- return G_t and objective J(theta).

    Eq. (1)  G_t = sum_{k=0}^{T-t-1} gamma^k R_{t+k+1}
    Eq. (2)  J(theta) = E_{pi_theta}[G_0]
    Eq. (3)  G_t = R_{t+1} + gamma G_{t+1},   G_T = 0

Convention: rewards[t] holds R_{t+1}, the reward of the transition (S_t, A_t).

Run:  python returns.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rlcore import Policy, TabularMDP, rollout


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    """Eq. (1), read forward: the k-th remaining reward is multiplied by gamma^k.

    k counts the distance from now, not the absolute time step.
    """
    return float(sum(gamma**k * r for k, r in enumerate(rewards)))


def returns_to_go(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """All of G_0, ..., G_{T-1} with the recursion of Eq. (3), computed backwards from G_T = 0.

    One backward pass costs O(T); applying Eq. (1) at every t would cost O(T^2).
    """
    T = len(rewards)
    G = np.zeros(T + 1)  # G[T] = 0 at the terminal
    for t in reversed(range(T)):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G[:T]


@dataclass(frozen=True)
class ObjectiveEstimate:
    """Monte Carlo estimate of J = E[G_0] from independent rollouts."""

    mean: float
    std_error: float
    returns: np.ndarray

    def __str__(self) -> str:
        return f"J ~= {self.mean:.3f} +- {self.std_error:.3f} (n={len(self.returns)})"


class ObjectiveEstimator:
    """Eq. (2): average G_0 over many rollouts of the same policy.

    One rollout gives one G_0; J is the expectation over the initial state,
    the sampled actions and the environment's randomness.
    """

    def __init__(self, env: TabularMDP, gamma: float | None = None, max_steps: int = 10_000):
        self.env = env
        self.gamma = env.gamma if gamma is None else gamma
        self.max_steps = max_steps

    def estimate(self, policy: Policy, n_episodes: int, rng: np.random.Generator) -> ObjectiveEstimate:
        returns = np.array([
            discounted_return(rollout(self.env, policy, rng, self.max_steps).rewards, self.gamma)
            for _ in range(n_episodes)
        ])
        se = returns.std(ddof=1) / np.sqrt(n_episodes) if n_episodes > 1 else float("nan")
        return ObjectiveEstimate(float(returns.mean()), float(se), returns)


if __name__ == "__main__":
    rewards, gamma = [1, 0, 4], 0.9
    print("Lecture example, rewards 1, 0, 4 and gamma = 0.9")
    print("  G_0 by Eq. (1):", round(discounted_return(rewards, gamma), 4))
    print("  G_t by Eq. (3):", np.round(returns_to_go(rewards, gamma), 4))
    print("Exercise 1, rewards 2, -1, 3 and gamma = 0.5")
    print("  G_0, G_1 =", np.round(returns_to_go([2, -1, 3], 0.5)[:2], 4))
