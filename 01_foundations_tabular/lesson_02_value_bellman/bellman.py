"""Lesson 2 -- value functions and the Bellman equations.

    Eq. (4)  v_pi(s) = E_pi[G_t | S_t = s],   q_pi(s, a) = E_pi[G_t | S_t = s, A_t = a]
    Eq. (5)  v_pi(s) = E_pi[R_{t+1} + gamma v_pi(S_{t+1}) | S_t = s]
    Eq. (6)  q_pi(s, a) = R^a_s + gamma sum_s' P^a_{ss'} v_pi(s')
    Eq. (7)  v_pi(s) = sum_a pi(a | s) q_pi(s, a)
    Eq. (8)  q_*(s, a) = R^a_s + gamma sum_s' P^a_{ss'} max_a' q_*(s', a')

Run:  python bellman.py
"""
from __future__ import annotations

import numpy as np

from rlcore import TabularMDP


class BellmanEquations:
    """The Bellman relations of one MDP, written with explicit sums so each line matches a formula."""

    def __init__(self, mdp: TabularMDP, gamma: float | None = None):
        self.mdp = mdp
        self.gamma = mdp.gamma if gamma is None else gamma

    def q_from_v(self, v: np.ndarray) -> np.ndarray:
        """Eq. (6): fix the first action a, then average v over the possible next states s'."""
        m, q = self.mdp, np.zeros((self.mdp.n_states, self.mdp.n_actions))
        for s in range(m.n_states):
            for a in range(m.n_actions):
                future = sum(m.P[s, a, s2] * (0.0 if m.terminal[s2] else v[s2]) for s2 in range(m.n_states))
                q[s, a] = m.R[s, a] + self.gamma * future
        return q

    @staticmethod
    def v_from_q(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Eq. (7): weighted average of the action values by the current policy (not the best action)."""
        return (pi * q).sum(axis=1)

    def expectation_backup(self, pi: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Right-hand side of Eq. (5) computed with the model: Eq. (6) followed by Eq. (7)."""
        v_new = self.v_from_q(pi, self.q_from_v(v))
        v_new[self.mdp.terminal] = 0.0
        return v_new

    def optimality_backup(self, q: np.ndarray) -> np.ndarray:
        """Right-hand side of Eq. (8): max over the next action, expectation over the next state."""
        return self.q_from_v(q.max(axis=1))

    def evaluate(self, pi: np.ndarray, tol: float = 1e-10, max_iters: int = 10_000) -> np.ndarray:
        """Find the v that satisfies Eq. (5) by repeating the backup (Lesson 3 studies this as DP)."""
        v = np.zeros(self.mdp.n_states)
        for _ in range(max_iters):
            v_new = self.expectation_backup(pi, v)
            if np.max(np.abs(v_new - v)) < tol:
                return v_new
            v = v_new
        return v

    def q_star(self, tol: float = 1e-10, max_iters: int = 10_000) -> np.ndarray:
        """Find q_* by repeating the optimality backup of Eq. (8)."""
        q = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        for _ in range(max_iters):
            q_new = self.optimality_backup(q)
            if np.max(np.abs(q_new - q)) < tol:
                return q_new
            q = q_new
        return q

    def sampled_target(self, pi: np.ndarray, v: np.ndarray, s: int, rng: np.random.Generator) -> float:
        """One transition gives ONE sample of R_{t+1} + gamma v(S_{t+1}), not the expectation of Eq. (5)."""
        m = self.mdp
        a = rng.choice(m.n_actions, p=pi[s])
        probs, next_states, rewards = zip(*m.outcomes[s][a])
        i = rng.choice(len(probs), p=probs)
        s2 = next_states[i]
        return rewards[i] + self.gamma * (0.0 if m.terminal[s2] else v[s2])


def lecture_example_mdp(v_u: float = 4.0, gamma: float = 0.9) -> TabularMDP:
    """Section 4 example.

    In s: action a pays 2, then goes to u with prob 0.75 or ends with prob 0.25;
          action b pays 3 and ends.
    In u: every action pays `v_u` and ends, so v_pi(u) = v_u for any policy.
    """
    return TabularMDP.from_transitions(
        states=["s", "u", "T"],
        actions=["a", "b"],
        transitions={
            ("s", "a"): [(0.75, "u", 2.0), (0.25, "T", 2.0)],
            ("s", "b"): [(1.0, "T", 3.0)],
            ("u", None): [(1.0, "T", v_u)],
        },
        terminal_states=["T"],
        start="s",
        gamma=gamma,
    )


def uniform_policy(mdp: TabularMDP) -> np.ndarray:
    return np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)


if __name__ == "__main__":
    for v_u in (4.0, 2.0):
        mdp = lecture_example_mdp(v_u)
        bell = BellmanEquations(mdp)
        pi = uniform_policy(mdp)
        v = bell.evaluate(pi)
        q = bell.q_from_v(v)
        s = mdp.state_index("s")
        print(f"v_pi(u) = {v_u}: q_pi(s,a) = {q[s, 0]:.3f}, q_pi(s,b) = {q[s, 1]:.3f}, v_pi(s) = {v[s]:.3f}")
