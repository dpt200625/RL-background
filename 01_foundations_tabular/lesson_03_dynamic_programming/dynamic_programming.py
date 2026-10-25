"""Lesson 3 -- dynamic programming and generalized policy iteration (GPI).

    Eq. (9)   V_{k+1}(s) = sum_a pi(a|s) [R^a_s + gamma sum_s' P^a_{ss'} V_k(s')]   policy evaluation
    Eq. (10)  pi'(s) in argmax_a q_pi(s, a)                                        policy improvement
    Eq. (11)  V_{k+1}(s) = max_a [R^a_s + gamma sum_s' P^a_{ss'} V_k(s')]           value iteration

k counts backup sweeps, not interaction steps. Every solver needs the model
(mdp.P, mdp.R); none of them samples a trajectory.

Run:  python dynamic_programming.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlcore import TabularMDP, TabularPolicy


@dataclass
class DPResult:
    v: np.ndarray
    policy: TabularPolicy
    sweeps: int                                        # total backup sweeps
    improvements: int = 0                              # policy-improvement steps (PI / GPI)
    deltas: list[float] = field(default_factory=list)  # ||V_{k+1} - V_k||_inf of every sweep


class BackupSweep:
    """One sweep of backups over all states, synchronous or in place.

    synchronous: every V_{k+1}(s) is computed from the old vector V_k.
    in place:    states are updated in order and later states already use
                 the new values (Exercise 2).
    """

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, in_place: bool = False):
        self.mdp = mdp
        self.gamma = mdp.gamma if gamma is None else gamma
        self.in_place = in_place

    def _q_row(self, s: int, v: np.ndarray) -> np.ndarray:
        """R^a_s + gamma sum_s' P^a_{ss'} V(s') for every a, with V(terminal) = 0."""
        return self.mdp.R[s] + self.gamma * self.mdp.P[s] @ np.where(self.mdp.terminal, 0.0, v)

    def __call__(self, v: np.ndarray, pi: np.ndarray | None = None) -> np.ndarray:
        """Expectation backup (Eq. 9) when `pi` is given, otherwise max backup (Eq. 11)."""
        mdp = self.mdp
        if not self.in_place:
            q = mdp.lookahead(v, self.gamma)
            v_new = q.max(axis=1) if pi is None else (pi * q).sum(axis=1)
            return np.where(mdp.terminal, 0.0, v_new)
        v = v.copy()
        for s in range(mdp.n_states):
            if mdp.terminal[s]:
                continue
            q_s = self._q_row(s, v)
            v[s] = q_s.max() if pi is None else pi[s] @ q_s
        return v


class PolicyEvaluation:
    """Repeat Eq. (9) with the policy held fixed until the value stops changing."""

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, theta: float = 1e-8,
                 in_place: bool = False, max_sweeps: int = 100_000):
        self.sweep = BackupSweep(mdp, gamma, in_place)
        self.theta, self.max_sweeps = theta, max_sweeps

    def run(self, pi: np.ndarray, v0: np.ndarray | None = None, n_sweeps: int | None = None):
        """Returns (V, deltas). `n_sweeps` caps the sweeps (truncated evaluation, used by GPI)."""
        v = np.zeros(self.sweep.mdp.n_states) if v0 is None else v0.copy()
        deltas = []
        limit = self.max_sweeps if n_sweeps is None else n_sweeps
        for _ in range(limit):
            v_new = self.sweep(v, pi)
            deltas.append(float(np.max(np.abs(v_new - v))))
            v = v_new
            if deltas[-1] < self.theta:
                break
        return v, deltas


class PolicyImprovement:
    """Eq. (10): act greedily w.r.t. q_pi = one-step lookahead on V, with a fixed tie rule."""

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, tie_break: str = "first"):
        self.mdp = mdp
        self.gamma = mdp.gamma if gamma is None else gamma
        self.tie_break = tie_break

    def __call__(self, v: np.ndarray) -> TabularPolicy:
        return TabularPolicy.greedy(self.mdp.lookahead(v, self.gamma), self.tie_break)


class GeneralizedPolicyIteration:
    """Alternate (possibly incomplete) evaluation and greedy improvement.

    eval_sweeps = None -> evaluate to convergence: classic policy iteration.
    eval_sweeps = k    -> only k evaluation sweeps per improvement (modified PI).
    Both steps act on the same V and policy until neither changes them.
    """

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, eval_sweeps: int | None = None,
                 theta: float = 1e-8, max_iterations: int = 10_000):
        self.mdp = mdp
        self.evaluation = PolicyEvaluation(mdp, gamma, theta)
        self.improvement = PolicyImprovement(mdp, gamma)
        self.eval_sweeps, self.theta, self.max_iterations = eval_sweeps, theta, max_iterations

    def run(self) -> DPResult:
        mdp = self.mdp
        policy = TabularPolicy.uniform(mdp.n_states, mdp.n_actions)
        v = np.zeros(mdp.n_states)
        deltas: list[float] = []
        for it in range(1, self.max_iterations + 1):
            v_new, d = self.evaluation.run(policy.table, v, self.eval_sweeps)
            deltas += d
            new_policy = self.improvement(v_new)
            value_settled = np.max(np.abs(v_new - v)) < self.theta
            v = v_new
            if np.array_equal(new_policy.table, policy.table) and (self.eval_sweeps is None or value_settled):
                return DPResult(v, policy, len(deltas), it, deltas)
            policy = new_policy
        return DPResult(v, policy, len(deltas), self.max_iterations, deltas)


class PolicyIteration(GeneralizedPolicyIteration):
    """Evaluate the policy completely, then improve it; stop when the greedy policy is stable."""

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, theta: float = 1e-8):
        super().__init__(mdp, gamma, eval_sweeps=None, theta=theta)


class ValueIteration:
    """Eq. (11): put the max directly inside the backup; extract the greedy policy at the end."""

    def __init__(self, mdp: TabularMDP, gamma: float | None = None, theta: float = 1e-8,
                 in_place: bool = False, max_sweeps: int = 100_000):
        self.mdp = mdp
        self.sweep = BackupSweep(mdp, gamma, in_place)
        self.improvement = PolicyImprovement(mdp, gamma)
        self.theta, self.max_sweeps = theta, max_sweeps

    def iterate(self, n_sweeps: int, v0: np.ndarray | None = None) -> list[np.ndarray]:
        """V_0, V_1, ..., V_n: handy to watch delayed reward travel backwards."""
        vs = [np.zeros(self.mdp.n_states) if v0 is None else v0.copy()]
        for _ in range(n_sweeps):
            vs.append(self.sweep(vs[-1]))
        return vs

    def run(self) -> DPResult:
        v = np.zeros(self.mdp.n_states)
        deltas = []
        for _ in range(self.max_sweeps):
            v_new = self.sweep(v)
            deltas.append(float(np.max(np.abs(v_new - v))))
            v = v_new
            if deltas[-1] < self.theta:
                break
        return DPResult(v, self.improvement(v), len(deltas), 0, deltas)


def contraction_ratios(deltas: list[float]) -> np.ndarray:
    """||V_{k+2} - V_{k+1}|| / ||V_{k+1} - V_k||; for gamma < 1 each ratio is at most gamma."""
    d = np.asarray(deltas)
    mask = d[:-1] > 1e-12
    return d[1:][mask] / d[:-1][mask]


def lecture_example_mdp(gamma: float = 0.9, u_reward: float = 4.0) -> TabularMDP:
    """Section 4 example: in u get 4 then terminal; in s, a pays 0 and goes to u, b pays 2 and ends.

    States are listed as (u, s) so that an in-place sweep updates u before s.
    `u_reward` lets Exercise 3 build a wrong model of the same task.
    """
    return TabularMDP.from_transitions(
        states=["u", "s", "T"],
        actions=["a", "b"],
        transitions={
            ("u", None): [(1.0, "T", u_reward)],
            ("s", "a"): [(1.0, "u", 0.0)],
            ("s", "b"): [(1.0, "T", 2.0)],
        },
        terminal_states=["T"],
        start="s",
        gamma=gamma,
    )


if __name__ == "__main__":
    mdp = lecture_example_mdp()
    s, u = mdp.state_index("s"), mdp.state_index("u")
    for k, v in enumerate(ValueIteration(mdp).iterate(2)):
        print(f"synchronous V_{k}: V(u) = {v[u]:.2f}, V(s) = {v[s]:.2f}")
    v1 = ValueIteration(mdp, in_place=True).iterate(1)[1]
    print(f"in-place sweep 1: V(u) = {v1[u]:.2f}, V(s) = {v1[s]:.2f}")
    for gamma in (0.9, 0.4):
        res = ValueIteration(lecture_example_mdp(gamma)).run()
        print(f"gamma = {gamma}: greedy action in s = {mdp.action_names[res.policy.table[s].argmax()]!r}")
