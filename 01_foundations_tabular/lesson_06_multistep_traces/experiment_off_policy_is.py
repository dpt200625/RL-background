"""Lesson 6 experiment -- ordinary vs weighted importance sampling.

5-state random walk (reward +1 on the right end, gamma = 1). Behavior mu
moves left/right with probability 0.5; the target pi moves right with
probability 0.75. The true J(pi) = v_pi(start) is computed from the model and
only used to measure the error of the two estimators.

Run:  python experiment_off_policy_is.py --runs 100 --episodes 300
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rlcore import RandomWalk, TabularPolicy, rollout, save_figure, save_json
from rlcore.plotting import plot_curves
from importance_sampling import OffPolicyMCEvaluator

ARTIFACT_DIR = ("01_foundations_tabular", "lesson_06_multistep_traces")


def make_policies(env: RandomWalk, p_right_target: float):
    behavior = TabularPolicy(env.uniform_policy())
    target_table = np.tile([1 - p_right_target, p_right_target], (env.n_states, 1))
    return TabularPolicy(target_table), behavior


def run(runs: int = 100, episodes: int = 300, p_right_target: float = 0.75, save: bool = True) -> dict:
    env = RandomWalk(n=5)
    target, behavior = make_policies(env, p_right_target)
    true_value = float(env.evaluate_policy_exact(target.table)[env.reset()])
    sq_err = {"ordinary IS": np.zeros((runs, episodes)), "weighted IS": np.zeros((runs, episodes))}
    all_weights = []
    for run in range(runs):
        env.seed(run)
        rng = np.random.default_rng(run)
        est = OffPolicyMCEvaluator(target, behavior)
        for ep in range(episodes):
            est.add(rollout(env, behavior, rng))
            sq_err["ordinary IS"][run, ep] = (est.ordinary - true_value) ** 2
            sq_err["weighted IS"][run, ep] = (est.weighted - true_value) ** 2
        all_weights += est.weights

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_curves(axes[0], sq_err, x=np.arange(1, episodes + 1), xlabel="behavior episodes",
                ylabel="squared error of J(pi) estimate", title=f"true J(pi) = {true_value:.3f}, {runs} runs")
    axes[0].set_xscale("log")
    w = np.array(all_weights)
    axes[1].hist(np.log10(w[w > 0]), bins=40)
    axes[1].set_xlabel("log10 W (trajectory importance weight)")
    axes[1].set_title(f"W ranges from {w.min():.3g} to {w.max():.3g}")
    fig.tight_layout()

    summary = {"true_value": true_value, "weight_max": float(w.max()), "weight_mean": float(w.mean()),
               **{name: {"mse_final": float(e[:, -1].mean())} for name, e in sq_err.items()}}
    if save:
        summary["figure"] = str(save_figure(fig, *ARTIFACT_DIR, "off_policy_is.png"))
        save_json(summary, *ARTIFACT_DIR, "off_policy_is.json")
    return {"sq_err": sq_err, "summary": summary, "figure": fig}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--p-right-target", type=float, default=0.75)
    args = parser.parse_args()
    out = run(args.runs, args.episodes, args.p_right_target)
    s = out["summary"]
    print(f"true J(pi) = {s['true_value']:.4f}; mean trajectory weight = {s['weight_mean']:.3f}, max = {s['weight_max']:.1f}")
    for name in ("ordinary IS", "weighted IS"):
        print(f"{name:12s} final MSE = {s[name]['mse_final']:.5f}")
    print("saved", s["figure"])


if __name__ == "__main__":
    main()
