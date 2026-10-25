"""Lesson 5 -- TD control: Sarsa, Expected Sarsa, Q-learning (and Double Q-learning).

All agents share one update rule
    Q(S_t, A_t) <- Q(S_t, A_t) + alpha [y - Q(S_t, A_t)]
and differ only in the target y:

    Eq. (15)  Sarsa           y = R_{t+1} + gamma Q(S_{t+1}, A_{t+1})        A_{t+1} actually sampled
    Eq. (16)  Expected Sarsa  y = R_{t+1} + gamma sum_a pi(a|S_{t+1}) Q(S_{t+1}, a)
    Eq. (17)  Q-learning      y = R_{t+1} + gamma max_a Q(S_{t+1}, a)

At a terminal S_{t+1} there is nothing to bootstrap: y = R_{t+1}.
The behavior policy mu is epsilon-greedy w.r.t. Q.

Run:  python td_control.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rlcore import EpsilonGreedyPolicy, TabularMDP, greedy_actions


class EpsilonSchedule:
    """epsilon as a function of the episode index k (1, 2, ...).

    constant(eps): fixed exploration -- not GLIE (Exercise 2).
    glie(c):       eps_k = min(1, c / k) -> 0, greedy in the limit.
    """

    def __init__(self, fn, name: str):
        self.fn, self.name = fn, name

    @classmethod
    def constant(cls, eps: float) -> "EpsilonSchedule":
        return cls(lambda k: eps, f"constant {eps}")

    @classmethod
    def glie(cls, c: float = 1.0) -> "EpsilonSchedule":
        return cls(lambda k: min(1.0, c / max(k, 1)), f"GLIE {c}/k")

    def __call__(self, k: int) -> float:
        return self.fn(k)


@dataclass
class EpisodeStats:
    total_reward: float
    length: int


class TDControlAgent(ABC):
    """Tabular TD control agent: epsilon-greedy behavior, one-pair update per transition."""

    #: Sarsa needs A_{t+1} before it can update; the others pick it after the update.
    needs_next_action = False

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5, gamma: float = 1.0,
                 epsilon: float | EpsilonSchedule = 0.1, seed: int | None = None):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.schedule = epsilon if isinstance(epsilon, EpsilonSchedule) else EpsilonSchedule.constant(epsilon)
        self.behavior = EpsilonGreedyPolicy(self.Q, self.schedule(1))
        self.rng = np.random.default_rng(seed)
        self.episodes = 0

    @property
    def epsilon(self) -> float:
        return self.behavior.epsilon

    def act(self, s: int) -> int:
        """A ~ mu(. | s): epsilon-greedy w.r.t. the current Q."""
        return self.behavior.sample(s, self.rng)

    def greedy_action(self, s: int) -> int:
        return int(greedy_actions(self.Q[s])[0])

    @abstractmethod
    def target(self, r: float, s_next: int, a_next: int | None, terminal: bool) -> float:
        ...

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int | None, terminal: bool) -> float:
        """Correct exactly one table entry; returns the TD error y - Q(s, a)."""
        delta = self.target(r, s_next, a_next, terminal) - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return delta

    def run_episode(self, env: TabularMDP, max_steps: int = 10_000) -> EpisodeStats:
        self.episodes += 1
        self.behavior.epsilon = self.schedule(self.episodes)
        s = env.reset()
        a = self.act(s)
        total, t = 0.0, 0
        for t in range(1, max_steps + 1):
            step = env.step(a)
            total += step.reward
            s_next = step.next_state
            if self.needs_next_action:
                a_next = None if step.terminated else self.act(s_next)
                self.update(s, a, step.reward, s_next, a_next, step.terminated)
            else:
                self.update(s, a, step.reward, s_next, None, step.terminated)
                a_next = None if step.terminated else self.act(s_next)
            if step.done:
                break
            s, a = s_next, a_next
        return EpisodeStats(total, t)

    def train(self, env: TabularMDP, episodes: int, max_steps: int = 10_000) -> np.ndarray:
        """Online sum of rewards of every training episode (exploration switched on)."""
        return np.array([self.run_episode(env, max_steps).total_reward for _ in range(episodes)])


class Sarsa(TDControlAgent):
    """On-policy: the target uses A_{t+1} that the behavior really samples, exploration included."""

    needs_next_action = True

    def target(self, r, s_next, a_next, terminal):
        return r if terminal else r + self.gamma * self.Q[s_next, a_next]


class ExpectedSarsa(TDControlAgent):
    """Average the next-action values under pi(. | S_{t+1}) instead of using one sample.

    The target policy pi defaults to the agent's own epsilon-greedy policy
    (on-policy Expected Sarsa); `target_epsilon` sets a different one.
    """

    def __init__(self, *args, target_epsilon: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_epsilon = target_epsilon

    def target_probs(self, s: int) -> np.ndarray:
        eps = self.epsilon if self.target_epsilon is None else self.target_epsilon
        return EpsilonGreedyPolicy(self.Q, eps).probs(s)

    def target(self, r, s_next, a_next, terminal):
        return r if terminal else r + self.gamma * self.target_probs(s_next) @ self.Q[s_next]


class QLearning(TDControlAgent):
    """Off-policy: the target assumes a greedy future action, whatever the behavior picks next."""

    def target(self, r, s_next, a_next, terminal):
        return r if terminal else r + self.gamma * self.Q[s_next].max()


class DoubleQLearning(TDControlAgent):
    """Two tables: one selects the argmax action, the other evaluates it (Sutton-Barto 6.7).

    Behavior is epsilon-greedy w.r.t. Q1 + Q2; each update changes one table at random.
    """

    def __init__(self, n_states: int, n_actions: int, *args, **kwargs):
        super().__init__(n_states, n_actions, *args, **kwargs)
        self.Q1, self.Q2 = np.zeros_like(self.Q), np.zeros_like(self.Q)

    def target(self, r, s_next, a_next, terminal):  # used with (select, evaluate) set by update()
        if terminal:
            return r
        a_star = greedy_actions(self._select[s_next])[0]
        return r + self.gamma * self._evaluate[s_next, a_star]

    def update(self, s, a, r, s_next, a_next, terminal):
        if self.rng.random() < 0.5:
            self._select, self._evaluate, table = self.Q1, self.Q2, self.Q1
        else:
            self._select, self._evaluate, table = self.Q2, self.Q1, self.Q2
        delta = self.target(r, s_next, a_next, terminal) - table[s, a]
        table[s, a] += self.alpha * delta
        self.Q[s] = self.Q1[s] + self.Q2[s]  # the behavior reads Q (in place, the policy keeps its reference)
        return delta


AGENTS = {"Sarsa": Sarsa, "Expected Sarsa": ExpectedSarsa, "Q-learning": QLearning, "Double Q-learning": DoubleQLearning}


def worked_example(pi_L: float, sampled: str, q_sa: float = 2.0, r: float = 1.0, gamma: float = 0.9,
                   alpha: float = 0.5, q_uL: float = 4.0, q_uR: float = 0.0) -> dict[str, tuple[float, float]]:
    """Section 4 / Exercise 1: the three targets and new estimates of Q(s, a).

    States: s = 0, u = 1. Actions: L = 0, R = 1. With two actions an
    epsilon-greedy policy gives pi(L | u) = 1 - eps / 2, so eps = 2 (1 - pi_L).
    """
    eps = 2.0 * (1.0 - pi_L)
    a_next = {"L": 0, "R": 1}[sampled]
    out = {}
    for name, cls in (("Sarsa", Sarsa), ("Expected Sarsa", ExpectedSarsa), ("Q-learning", QLearning)):
        agent = cls(2, 2, alpha=alpha, gamma=gamma, epsilon=eps)
        agent.Q[:] = [[q_sa, 0.0], [q_uL, q_uR]]
        y = agent.target(r, 1, a_next, False)
        agent.update(0, 0, r, 1, a_next, False)
        out[name] = (round(float(y), 4), round(float(agent.Q[0, 0]), 4))
    return out


if __name__ == "__main__":
    print("Section 4 (pi(L|u) = 0.75, sampled R):", worked_example(0.75, "R"))
    print("Exercise 1 (pi(L|u) = 0.5, sampled L):", worked_example(0.5, "L"))
