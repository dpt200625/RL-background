"""Lesson 5 -- maximization bias, and how a double estimator reduces it.

Every action has true value 0, so max_a q(a) = 0. Each estimate Q(a) is a
sample mean of noisy rewards. Taking max_a Q(a) uses the same noise to pick
the action and to score it, so on average it overestimates (bias > 0).
The double estimator picks the action with one half of the data and scores it
with the other half, separating selection from evaluation (the idea behind
Double Q-learning). It removes this source of bias here, but it does not
remove every estimation error: its own variance stays.

Run:  python maximization_bias.py
"""
from __future__ import annotations

import numpy as np


def single_and_double_estimates(n_actions: int, n_samples: int, n_trials: int, noise_std: float = 1.0,
                                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (max_a Q(a), Q_B(argmax_a Q_A(a))) for `n_trials` independent data sets."""
    rng = np.random.default_rng(seed)
    rewards = rng.normal(0.0, noise_std, size=(n_trials, n_actions, 2 * n_samples))
    q_all = rewards.mean(axis=2)
    q_a, q_b = rewards[..., :n_samples].mean(axis=2), rewards[..., n_samples:].mean(axis=2)
    single = q_all.max(axis=1)
    double = q_b[np.arange(n_trials), q_a.argmax(axis=1)]
    return single, double


if __name__ == "__main__":
    print("true max value = 0")
    for n_actions in (2, 10, 50):
        single, double = single_and_double_estimates(n_actions, n_samples=5, n_trials=20_000)
        print(f"{n_actions:3d} actions: E[max Q] = {single.mean():+.3f}   double estimator = {double.mean():+.3f}")
