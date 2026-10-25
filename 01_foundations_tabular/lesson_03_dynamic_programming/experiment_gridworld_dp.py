"""Lesson 3 experiment -- planning on a grid with a known map and known (slippery) transitions.

Compares policy iteration, value iteration and GPI with k evaluation sweeps:
* all of them reach the same optimal values and greedy policy,
* ||V_{k+1} - V_k||_inf shrinks at least as fast as gamma^k (contraction).

Outputs go to artifacts/01_foundations_tabular/lesson_03_dynamic_programming/.

Run:  python experiment_gridworld_dp.py --slip 0.1 --gamma 0.95
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from rlcore import GridWorld, save_figure, save_json
from rlcore.plotting import plot_grid_values
from dynamic_programming import GeneralizedPolicyIteration, PolicyIteration, ValueIteration

ARTIFACT_DIR = ("01_foundations_tabular", "lesson_03_dynamic_programming")

LAYOUT = ["S.....",
          ".##.#.",
          "...#..",
          ".#...#",
          ".#.#..",
          "...#.G"]


def run(slip: float = 0.1, gamma: float = 0.95, save: bool = True) -> dict:
    env = GridWorld(LAYOUT, slip=slip, gamma=gamma)
    solvers = {
        "policy iteration": PolicyIteration(env),
        "value iteration": ValueIteration(env),
        "GPI (k=3 eval sweeps)": GeneralizedPolicyIteration(env, eval_sweeps=3),
        "value iteration, in place": ValueIteration(env, in_place=True),
    }
    results, summary = {}, {}
    for name, solver in solvers.items():
        t0 = time.perf_counter()
        res = solver.run()
        results[name] = res
        summary[name] = {"sweeps": res.sweeps, "improvements": res.improvements,
                         "seconds": time.perf_counter() - t0, "V(start)": float(res.v[env.start_state])}

    ref = results["policy iteration"].v
    for name, res in results.items():
        summary[name]["max |V - V_PI|"] = float(np.max(np.abs(res.v - ref)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    best = results["value iteration"]
    im = plot_grid_values(axes[0], env, best.v, f"V* and greedy policy (slip={slip}, gamma={gamma})", best.policy.table)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    # policy iteration restarts its evaluation after every improvement, so its deltas are not one curve
    pi = results["policy iteration"]
    for name, res in results.items():
        if name != "policy iteration":
            axes[1].semilogy(np.maximum(res.deltas, 1e-12), label=f"{name} ({res.sweeps} sweeps)")
    k = np.arange(results["value iteration"].sweeps)
    axes[1].semilogy(k, results["value iteration"].deltas[0] * gamma**k, "k--", label=r"$\gamma^k$ reference")
    axes[1].set_xlabel("backup sweep k")
    axes[1].set_ylabel(r"$\|V_{k+1}-V_k\|_\infty$")
    axes[1].set_title(f"Convergence (policy iteration: {pi.sweeps} sweeps, {pi.improvements} improvements)",
                      fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    if save:
        summary["figure"] = str(save_figure(fig, *ARTIFACT_DIR, "gridworld_dp.png"))
        save_json(summary, *ARTIFACT_DIR, "gridworld_dp.json")
    return {"env": env, "results": results, "summary": summary, "figure": fig}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slip", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()
    out = run(args.slip, args.gamma)
    print(out["env"].render_policy(out["results"]["policy iteration"].policy.table))
    for name, row in out["summary"].items():
        if isinstance(row, dict):
            print(f"{name:28s} sweeps={row['sweeps']:4d} improvements={row['improvements']:3d} "
                  f"V(start)={row['V(start)']:.3f} max|V-V_PI|={row['max |V - V_PI|']:.1e}")
    print("saved", out["summary"]["figure"])


if __name__ == "__main__":
    main()
