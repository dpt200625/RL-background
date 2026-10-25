"""Lesson 4 experiment -- MC vs TD(0) prediction on the 5-state random walk (Sutton-Barto Example 6.2).

The uniform policy is fixed; the true values 1/6, ..., 5/6 come from solving
the Bellman equation with the known model and are only used to measure error.
The learners never read the model.

Plots
  left : TD(0) estimates after 0, 1, 10, 100 episodes vs the true values,
  right: RMS error over states, averaged over runs, for several step sizes.

Run:  python experiment_random_walk.py --runs 100 --episodes 100
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rlcore import RandomWalk, TabularPolicy, rollout, save_figure, save_json
from rlcore.plotting import plot_curves
from prediction import MonteCarloPrediction, TD0Prediction, ValuePredictor

ARTIFACT_DIR = ("01_foundations_tabular", "lesson_04_mc_td_prediction")


def make_learner(kind: str, env: RandomWalk, alpha: float) -> ValuePredictor:
    common = dict(n_states=env.n_states, gamma=1.0, init_value=0.5, terminal=env.terminal)
    if kind == "TD":
        return TD0Prediction(alpha=alpha, **common)
    return MonteCarloPrediction(alpha=alpha, **common)


def rms_curve(kind: str, alpha: float, runs: int, episodes: int, seed: int = 0) -> np.ndarray:
    """RMS error after each episode, shape (runs, episodes + 1)."""
    env = RandomWalk(n=5)
    policy = TabularPolicy(env.uniform_policy())
    true_v = env.true_values()
    errors = np.zeros((runs, episodes + 1))
    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        env.seed(seed + run)
        learner = make_learner(kind, env, alpha)
        errors[run, 0] = np.sqrt(np.mean((learner.V[1:-1] - true_v) ** 2))
        for ep in range(episodes):
            learner.learn_episode(rollout(env, policy, rng))
            errors[run, ep + 1] = np.sqrt(np.mean((learner.V[1:-1] - true_v) ** 2))
    return errors


def td_snapshots(alpha: float = 0.1, checkpoints=(0, 1, 10, 100), seed: int = 0) -> dict[int, np.ndarray]:
    env = RandomWalk(n=5, seed=seed)
    policy = TabularPolicy(env.uniform_policy())
    rng = np.random.default_rng(seed)
    learner = make_learner("TD", env, alpha)
    snaps = {0: learner.V[1:-1].copy()}
    for ep in range(1, max(checkpoints) + 1):
        learner.learn_episode(rollout(env, policy, rng))
        if ep in checkpoints:
            snaps[ep] = learner.V[1:-1].copy()
    return snaps


def run(runs: int = 100, episodes: int = 100, save: bool = True) -> dict:
    configs = [("TD", a) for a in (0.05, 0.1, 0.15)] + [("MC", a) for a in (0.01, 0.02, 0.03, 0.04)]
    curves = {f"{kind} alpha={a}": rms_curve(kind, a, runs, episodes) for kind, a in configs}

    env = RandomWalk(n=5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ep, v in td_snapshots().items():
        axes[0].plot(env.state_names[1:-1], v, marker="o", label=f"after {ep} episodes")
    axes[0].plot(env.state_names[1:-1], env.true_values(), "k--", label="true values")
    axes[0].set_title("TD(0) estimates, alpha = 0.1")
    axes[0].set_xlabel("state")
    axes[0].legend(fontsize=8)
    plot_curves(axes[1], curves, xlabel="episodes", ylabel="RMS error (averaged over states)",
                title=f"MC vs TD(0), {runs} runs")
    fig.tight_layout()

    summary = {name: {"rms_after_10": float(c[:, 10].mean()), "rms_final": float(c[:, -1].mean())}
               for name, c in curves.items()}
    if save:
        summary["figure"] = str(save_figure(fig, *ARTIFACT_DIR, "random_walk_mc_vs_td.png"))
        save_json(summary, *ARTIFACT_DIR, "random_walk_mc_vs_td.json")
    return {"curves": curves, "summary": summary, "figure": fig}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    out = run(args.runs, args.episodes)
    for name, row in out["summary"].items():
        if isinstance(row, dict):
            print(f"{name:16s} RMS after 10 episodes = {row['rms_after_10']:.3f}, final = {row['rms_final']:.3f}")
    print("saved", out["summary"]["figure"])


if __name__ == "__main__":
    main()
