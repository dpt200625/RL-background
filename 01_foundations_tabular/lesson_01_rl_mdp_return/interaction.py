"""Lesson 1 -- the agent-environment loop and the pipeline data -> learning -> use and evaluation.

* AgentEnvironmentLoop records transitions (S_t, A_t, R_{t+1}, S_{t+1}), the input of online RL.
* delayed_reward_mdp() is the lecture example: rewards 1, 0, 4 versus an action
  that ends at once with reward 3.
* A time limit produces `truncated`, which is not the task's terminal (Exercise 3).

Run:  python interaction.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlcore import Policy, TabularMDP, TabularPolicy, Trajectory, evaluate_policy, rollout


@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float      # R_{t+1}
    next_state: int
    terminated: bool   # S_{t+1} is a terminal of the task: v(S_{t+1}) = 0
    truncated: bool    # cut by a time limit: the task itself goes on


class AgentEnvironmentLoop:
    """Step 1 of the pipeline (data): the agent picks A_t, the environment returns R_{t+1}, S_{t+1}."""

    def __init__(self, env: TabularMDP, policy: Policy, rng: np.random.Generator):
        self.env, self.policy, self.rng = env, policy, rng

    def run_episode(self, max_steps: int = 10_000) -> list[Transition]:
        transitions = []
        s = self.env.reset()
        for _ in range(max_steps):
            a = self.policy.sample(s, self.rng)
            step = self.env.step(a)
            transitions.append(Transition(s, a, step.reward, step.next_state, step.terminated, step.truncated))
            if step.done:
                break
            s = step.next_state
        return transitions


def delayed_reward_mdp(gamma: float = 0.9) -> TabularMDP:
    """Lecture example: 'go' earns 1, 0, 4 over three steps; 'stop' earns 3 and ends at once."""
    return TabularMDP.from_transitions(
        states=["s0", "s1", "s2", "end"],
        actions=["go", "stop"],
        transitions={
            ("s0", "go"): [(1.0, "s1", 1.0)],
            ("s0", "stop"): [(1.0, "end", 3.0)],
            ("s1", None): [(1.0, "s2", 0.0)],
            ("s2", None): [(1.0, "end", 4.0)],
        },
        terminal_states=["end"],
        start="s0",
        gamma=gamma,
    )


def first_action_policy(mdp: TabularMDP, first_action: str) -> TabularPolicy:
    """Take `first_action` in s0, then 'go' (all later states behave the same for both actions)."""
    actions = np.full(mdp.n_states, mdp.action_index("go"))
    actions[mdp.state_index("s0")] = mdp.action_index(first_action)
    return TabularPolicy.deterministic(actions, mdp.n_actions)


def compare_first_actions(gamma: float = 0.9) -> dict[str, dict[str, float]]:
    """Immediate reward vs return for the two first actions of the lecture example."""
    mdp = delayed_reward_mdp(gamma)
    rng = np.random.default_rng(0)
    out = {}
    for name in ("go", "stop"):
        traj = rollout(mdp, first_action_policy(mdp, name), rng)
        out[name] = {"immediate_reward": traj.rewards[0], "return_G0": traj.discounted_return(gamma)}
    return out


def pipeline_demo(env: TabularMDP, policy: Policy, n_eval: int, seed: int = 0) -> dict[str, float]:
    """Steps 1 and 3 of the pipeline. Step 2 (learning) is what Lessons 2-6 add.

    Training data and evaluation rollouts use separate random streams.
    """
    data_rng, eval_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    data = AgentEnvironmentLoop(env, policy, data_rng).run_episode()
    returns = evaluate_policy(env, policy, n_eval, eval_rng)
    return {"collected_transitions": len(data), "eval_mean_return": float(returns.mean()),
            "eval_std_return": float(returns.std())}


def truncated_rollout(env: TabularMDP, policy: Policy, time_limit: int, seed: int = 0) -> Trajectory:
    """Roll out with a time limit: the episode may end as truncated although no terminal was reached."""
    old = env.max_steps
    env.max_steps = time_limit
    try:
        return rollout(env, policy, np.random.default_rng(seed))
    finally:
        env.max_steps = old


if __name__ == "__main__":
    for action, res in compare_first_actions().items():
        print(f"first action {action!r}: immediate reward {res['immediate_reward']:.1f}, return {res['return_G0']:.2f}")
