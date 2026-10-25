import numpy as np
import pytest

from rlcore import (EpsilonGreedyPolicy, GridWorld, RandomWalk, TabularMDP, TabularPolicy, cliff_walking,
                    evaluate_policy, rollout)


def test_random_walk_true_values():
    np.testing.assert_allclose(RandomWalk(n=5).true_values(), np.arange(1, 6) / 6)
    np.testing.assert_allclose(RandomWalk(n=19, left_reward=-1).true_values(), np.linspace(-0.9, 0.9, 19), atol=1e-12)


def test_transition_model_is_a_distribution():
    env = GridWorld(["S..", ".#.", "..G"], slip=0.2)
    np.testing.assert_allclose(env.P.sum(axis=2), 1.0)
    assert env.terminal[env.to_state(2, 2)]


def test_cliff_sends_back_to_start_without_ending():
    env = cliff_walking()
    env.reset()
    step = env.step(1)  # right from S is the cliff
    assert step.next_state == env.start_state and step.reward == -100 and not step.terminated


def test_time_limit_is_truncation_not_termination():
    env = GridWorld(["S...G"], max_steps=2)
    traj = rollout(env, TabularPolicy.deterministic(np.full(env.n_states, 3), 4), np.random.default_rng(0))
    assert traj.truncated and not traj.terminated and len(traj) == 2


def test_exact_evaluation_matches_rollouts():
    env = RandomWalk(n=5, seed=0)
    pi = TabularPolicy(env.uniform_policy())
    est = evaluate_policy(env, pi, 4000, np.random.default_rng(0)).mean()
    assert abs(est - 0.5) < 0.03


def test_epsilon_greedy_probabilities():
    q = np.array([[1.0, 0.0, 1.0]])
    p = EpsilonGreedyPolicy(q, 0.3).probs(0)
    np.testing.assert_allclose(p, [0.1 + 0.35, 0.1, 0.1 + 0.35])


def test_from_transitions_rejects_missing_actions():
    with pytest.raises(ValueError):
        TabularMDP.from_transitions(["s", "T"], ["a", "b"], {("s", "a"): [(1.0, "T", 0.0)]}, ["T"], "s")
