"""Lesson 6 -- off-policy estimation with importance sampling.

    Eq. (21)  rho_t = pi(A_t | S_t) / mu(A_t | S_t)

The data come from the behavior mu, we want E_pi[G_0] of the target pi.
Multiplying the ratios along a trajectory re-weights the whole trajectory:

    ordinary IS:  V = (1/N) sum_i W_i G_i                (unbiased, variance can be huge)
    weighted IS:  V = sum_i W_i G_i / sum_i W_i          (biased for finite N, usually lower variance)

with W_i = prod_t rho_t of episode i. If pi needs an action that mu never
takes (mu = 0), the ratio is undefined: there are no samples of that action.

Run:  python importance_sampling.py
"""
from __future__ import annotations

import numpy as np

from rlcore import Policy, Trajectory


class SupportError(ValueError):
    """The target policy needs an action that the behavior policy never takes."""


def importance_ratio(pi_prob: float, mu_prob: float) -> float:
    """Eq. (21) for one step. Refuses mu = 0: division cannot create the missing data."""
    if mu_prob <= 0.0:
        raise SupportError(f"behavior probability is {mu_prob}; target probability {pi_prob} has no support in the data")
    return pi_prob / mu_prob


def trajectory_weight(traj: Trajectory, target: Policy, behavior: Policy) -> float:
    """W = prod_t pi(A_t|S_t) / mu(A_t|S_t) over one episode."""
    w = 1.0
    for s, a in zip(traj.states, traj.actions):
        w *= importance_ratio(target.probs(s)[a], behavior.probs(s)[a])
    return w


class OffPolicyMCEvaluator:
    """Estimate J(pi) = E_pi[G_0] from complete episodes of mu with ordinary and weighted IS."""

    def __init__(self, target: Policy, behavior: Policy, gamma: float = 1.0):
        self.target, self.behavior, self.gamma = target, behavior, gamma
        self.sum_wg = 0.0
        self.sum_w = 0.0
        self.n = 0
        self.weights: list[float] = []

    def add(self, traj: Trajectory) -> None:
        w = trajectory_weight(traj, self.target, self.behavior)
        g = traj.discounted_return(self.gamma)
        self.sum_wg += w * g
        self.sum_w += w
        self.n += 1
        self.weights.append(w)

    @property
    def ordinary(self) -> float:
        return self.sum_wg / self.n if self.n else float("nan")

    @property
    def weighted(self) -> float:
        return self.sum_wg / self.sum_w if self.sum_w > 0 else 0.0


if __name__ == "__main__":
    print("ratio 0.8 / 0.4 =", importance_ratio(0.8, 0.4))
    try:
        importance_ratio(0.2, 0.0)
    except SupportError as err:
        print("support violated:", err)
