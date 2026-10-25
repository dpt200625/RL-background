"""Lesson 6 experiment -- credit assignment over several steps on the 19-state random walk.

Sutton-Barto Example 7.1 / Figure 12.3 setting: reward -1 on the left end,
+1 on the right end, gamma = 1, uniform policy. For each step size alpha we
report the RMS error after the first `episodes` episodes, averaged over states
and runs. All configurations reuse the same episodes (common random numbers),
so differences come from the update rule, not from the data.

Run:  python experiment_nstep_lambda.py --runs 20 --episodes 10
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from rlcore import RandomWalk, TabularPolicy, rollout, save_figure, save_json
from multistep import NStepTDPrediction, TDLambdaPrediction

ARTIFACT_DIR = ("01_foundations_tabular", "lesson_06_multistep_traces")


def sample_episodes(runs: int, episodes: int, seed: int = 0):
    env = RandomWalk(n=19, left_reward=-1.0)
    policy = TabularPolicy(env.uniform_policy())
    data = []
    for run in range(runs):
        env.seed(seed + run)
        rng = np.random.default_rng(seed + run)
        data.append([rollout(env, policy, rng) for _ in range(episodes)])
    return env, data


def rms_after_training(make_learner, env: RandomWalk, data) -> float:
    true_v = env.true_values()
    errs = []
    for episodes in data:
        learner = make_learner()
        for traj in episodes:
            learner.learn_episode(traj)
        errs.append(np.sqrt(np.mean((learner.V[1:-1] - true_v) ** 2)))
    return float(np.mean(errs))


def run(runs: int = 20, episodes: int = 10, ns=(1, 2, 4, 8, 16), lams=(0.0, 0.4, 0.8, 0.9),
        alphas=np.linspace(0.0, 1.0, 11), save: bool = True) -> dict:
    env, data = sample_episodes(runs, episodes)
    S = env.n_states
    nstep = {n: [rms_after_training(lambda: NStepTDPrediction(S, n, a), env, data) for a in alphas] for n in ns}
    tdlam = {lam: [rms_after_training(lambda: TDLambdaPrediction(S, lam, a), env, data) for a in alphas] for lam in lams}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for n, errs in nstep.items():
        axes[0].plot(alphas, errs, marker=".", label=f"n = {n}")
    for lam, errs in tdlam.items():
        axes[1].plot(alphas, errs, marker=".", label=f"lambda = {lam}")
    axes[0].set_title("n-step TD")
    axes[1].set_title("online TD(lambda), accumulating traces")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.set_ylim(0.2, 0.6)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"RMS error after {episodes} episodes ({runs} runs)")
    fig.tight_layout()

    best = lambda curves: {str(k): {"best_alpha": round(float(alphas[int(np.argmin(v))]), 2), "best_rms": float(np.min(v))}
                           for k, v in curves.items()}
    summary = {"n_step": best(nstep), "td_lambda": best(tdlam)}
    if save:
        summary["figure"] = str(save_figure(fig, *ARTIFACT_DIR, "random_walk19_nstep_lambda.png"))
        save_json(summary, *ARTIFACT_DIR, "random_walk19_nstep_lambda.json")
    return {"nstep": nstep, "tdlam": tdlam, "summary": summary, "figure": fig}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    out = run(args.runs, args.episodes)
    for family in ("n_step", "td_lambda"):
        for k, row in out["summary"][family].items():
            print(f"{family:9s} {k:>4s}: best alpha = {row['best_alpha']:.1f}, RMS = {row['best_rms']:.3f}")
    print("saved", out["summary"]["figure"])


if __name__ == "__main__":
    main()
