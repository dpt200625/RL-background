"""Agent-environment interaction: collect trajectories and evaluate a policy by rollouts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .mdp import TabularMDP
from .policies import Policy


@dataclass
class Trajectory:
    """One episode S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T.

    rewards[t] stores R_{t+1}, the reward produced by the transition (S_t, A_t).
    states has one more entry than actions: the last state S_T.
    """

    states: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.actions)

    def transitions(self) -> Iterator[tuple[int, int, float, int, bool]]:
        """Yield (S_t, A_t, R_{t+1}, S_{t+1}, S_{t+1} is terminal)."""
        T = len(self.actions)
        for t in range(T):
            last = t == T - 1
            yield self.states[t], self.actions[t], self.rewards[t], self.states[t + 1], last and self.terminated

    def discounted_return(self, gamma: float) -> float:
        """G_0 of this episode."""
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
        return g


def rollout(env: TabularMDP, policy: Policy, rng: np.random.Generator, max_steps: int = 10_000) -> Trajectory:
    """Run one episode of `policy` in `env` (stops at a terminal, a time limit or max_steps)."""
    traj = Trajectory(states=[env.reset()])
    s = traj.states[0]
    for _ in range(max_steps):
        a = policy.sample(s, rng)
        step = env.step(a)
        traj.actions.append(a)
        traj.rewards.append(step.reward)
        traj.states.append(step.next_state)
        s = step.next_state
        if step.done:
            traj.terminated, traj.truncated = step.terminated, step.truncated
            break
    else:
        traj.truncated = True
    return traj


def evaluate_policy(env: TabularMDP, policy: Policy, n_episodes: int, rng: np.random.Generator,
                    gamma: float | None = None, max_steps: int = 10_000) -> np.ndarray:
    """Returns G_0 of `n_episodes` independent rollouts. Their mean estimates J = E[G_0] (Eq. 2)."""
    gamma = env.gamma if gamma is None else gamma
    return np.array([rollout(env, policy, rng, max_steps).discounted_return(gamma) for _ in range(n_episodes)])
