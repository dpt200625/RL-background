"""Lesson 5 experiment -- Cliff Walking (Sutton-Barto Example 6.6).

Every step costs -1; stepping into the cliff costs -100 and sends the agent
back to the start. With a fixed epsilon = 0.1:
* Sarsa evaluates the exploring behavior, so it learns a path away from the edge;
* Q-learning learns the values of the greedy target, the path along the edge,
  but its online reward suffers because exploration sometimes falls off.

Two evaluations are reported separately: online reward with exploration on,
and the return of the greedy policy (exploration off, ties -> first action).

Run:  python experiment_cliff_walking.py --runs 20 --episodes 500
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rlcore import TabularPolicy, cliff_walking, rollout, save_figure, save_json
from rlcore.plotting import moving_average, plot_curves
from td_control import ExpectedSarsa, QLearning, Sarsa

ARTIFACT_DIR = ("01_foundations_tabular", "lesson_05_sarsa_qlearning")
AGENTS = {"Sarsa": Sarsa, "Expected Sarsa": ExpectedSarsa, "Q-learning": QLearning}


def greedy_return(env, q: np.ndarray, max_steps: int = 200) -> float:
    """Deployment check: follow argmax Q with exploration off (capped in case of loops)."""
    policy = TabularPolicy.greedy(q)
    traj = rollout(env, policy, np.random.default_rng(0), max_steps)
    return traj.discounted_return(1.0) if traj.terminated else float("nan")


def run(runs: int = 20, episodes: int = 500, alpha: float = 0.5, epsilon: float = 0.1,
        smooth: int = 10, save: bool = True) -> dict:
    env = cliff_walking()
    online, greedy, last_q = {}, {}, {}
    for name, cls in AGENTS.items():
        rewards = np.zeros((runs, episodes))
        greedy[name] = []
        for run in range(runs):
            env.seed(run)
            agent = cls(env.n_states, env.n_actions, alpha=alpha, gamma=1.0, epsilon=epsilon, seed=run)
            rewards[run] = agent.train(env, episodes)
            greedy[name].append(greedy_return(env, agent.Q))
        online[name] = rewards
        last_q[name] = agent.Q.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    plot_curves(axes[0], {k: moving_average(v, smooth) for k, v in online.items()},
                xlabel=f"episode (moving average of {smooth})", ylabel="sum of rewards per episode",
                title=f"Online reward, epsilon = {epsilon}, {runs} runs")
    axes[0].set_ylim(-100, 0)
    axes[1].imshow(np.zeros((env.n_rows, env.n_cols)), cmap="Greys", vmin=0, vmax=1)
    for c in range(1, env.n_cols - 1):
        axes[1].add_patch(plt.Rectangle((c - 0.5, env.n_rows - 1.5), 1, 1, color="tab:red", alpha=0.3))
    for (name, q), offset in zip(last_q.items(), (-0.15, 0.0, 0.15)):
        path = env.greedy_path(TabularPolicy.greedy(q).table)
        rows, cols = zip(*(env.to_pos(s) for s in path))
        axes[1].plot(np.array(cols), np.array(rows) + offset, marker=".", label=name)
    axes[1].set_title("Greedy path learned (last run); red = cliff")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.tight_layout()

    summary = {name: {"online_reward_last_100": float(online[name][:, -100:].mean()),
                      "greedy_return_mean": float(np.nanmean(greedy[name])),
                      "greedy_reached_goal": int(np.sum(~np.isnan(greedy[name])))}
               for name in AGENTS}
    if save:
        summary["figure"] = str(save_figure(fig, *ARTIFACT_DIR, "cliff_walking.png"))
        save_json(summary, *ARTIFACT_DIR, "cliff_walking.json")
    return {"env": env, "online": online, "q": last_q, "summary": summary, "figure": fig}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    out = run(args.runs, args.episodes, args.alpha, args.epsilon)
    for name, row in out["summary"].items():
        if isinstance(row, dict):
            print(f"{name:15s} online reward (last 100 ep) = {row['online_reward_last_100']:7.1f}   "
                  f"greedy return = {row['greedy_return_mean']:6.1f} ({row['greedy_reached_goal']} runs reached G)")
    for name, q in out["q"].items():
        print(f"\n{name} greedy policy:\n{out['env'].render_policy(TabularPolicy.greedy(q).table)}")
    print("\nsaved", out["summary"]["figure"])


if __name__ == "__main__":
    main()
