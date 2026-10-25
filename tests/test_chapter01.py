"""Chapter I: every worked example and exercise answer of the lecture, recomputed by the lesson code."""
import numpy as np
import pytest

from rlcore import RandomWalk, TabularPolicy, Trajectory, cliff_walking, rollout

CH = "01_foundations_tabular"


def test_lesson01_returns(lesson):
    m = lesson(f"{CH}/lesson_01_rl_mdp_return", "returns")
    np.testing.assert_allclose(m.returns_to_go([1, 0, 4], 0.9), [4.24, 3.6, 4.0])
    assert m.discounted_return([1, 0, 4], 0.9) == pytest.approx(4.24)
    np.testing.assert_allclose(m.returns_to_go([2, -1, 3], 0.5)[:2], [2.25, 0.5])  # exercise 1
    assert 1 + 0.9 * 4 == pytest.approx(m.returns_to_go([1, 4], 0.9)[0])            # G_t = 4.6


def test_lesson01_delayed_reward(lesson):
    m = lesson(f"{CH}/lesson_01_rl_mdp_return", "interaction")
    res = m.compare_first_actions()
    assert res["stop"]["immediate_reward"] > res["go"]["immediate_reward"]
    assert res["go"]["return_G0"] == pytest.approx(4.24) and res["stop"]["return_G0"] == pytest.approx(3.0)


def test_lesson01_single_frame_is_not_markov(lesson):
    m = lesson(f"{CH}/lesson_01_rl_mdp_return", "observability")
    one = m.predictability(m.next_observation_table(m.MovingObject(seed=0), 1))
    two = m.predictability(m.next_observation_table(m.MovingObject(seed=0), 2))
    assert two > one + 0.2


@pytest.mark.parametrize("v_u, q_sa, v_s", [(4.0, 4.7, 3.85), (2.0, 3.35, 3.175)])
def test_lesson02_bellman_example(lesson, v_u, q_sa, v_s):
    m = lesson(f"{CH}/lesson_02_value_bellman", "bellman")
    mdp = m.lecture_example_mdp(v_u)
    bell = m.BellmanEquations(mdp)
    pi = m.uniform_policy(mdp)
    v = bell.evaluate(pi)
    q = bell.q_from_v(v)
    s = mdp.state_index("s")
    assert q[s, 0] == pytest.approx(q_sa) and q[s, 1] == pytest.approx(3.0) and v[s] == pytest.approx(v_s)
    np.testing.assert_allclose(v, mdp.evaluate_policy_exact(pi))
    np.testing.assert_allclose(bell.q_from_v(v), mdp.lookahead(v))


def test_lesson03_dp_example(lesson):
    m = lesson(f"{CH}/lesson_03_dynamic_programming", "dynamic_programming")
    mdp = m.lecture_example_mdp(0.9)
    s, u = mdp.state_index("s"), mdp.state_index("u")
    v0, v1, v2 = m.ValueIteration(mdp).iterate(2)
    assert (v1[u], v1[s], v2[s]) == pytest.approx((4.0, 2.0, 3.6))
    assert m.ValueIteration(mdp, in_place=True).iterate(1)[1][s] == pytest.approx(3.6)   # exercise 2
    assert m.ValueIteration(m.lecture_example_mdp(0.4)).run().policy.table[s].argmax() == 1  # exercise 1: b


def test_lesson03_solvers_agree_and_contract(lesson):
    m = lesson(f"{CH}/lesson_03_dynamic_programming", "dynamic_programming")
    exp = lesson(f"{CH}/lesson_03_dynamic_programming", "experiment_gridworld_dp")
    from rlcore import GridWorld
    env = GridWorld(exp.LAYOUT, slip=0.1, gamma=0.9)
    pi = m.PolicyIteration(env).run()
    vi = m.ValueIteration(env).run()
    gpi = m.GeneralizedPolicyIteration(env, eval_sweeps=2).run()
    np.testing.assert_allclose(vi.v, pi.v, atol=1e-6)
    np.testing.assert_allclose(gpi.v, pi.v, atol=1e-6)
    assert np.all(m.contraction_ratios(vi.deltas) <= 0.9 + 1e-9)


def test_lesson04_examples(lesson):
    m = lesson(f"{CH}/lesson_04_mc_td_prediction", "prediction")
    td = m.TD0Prediction(2, gamma=0.9, alpha=0.5)
    td.V[:] = [1.0, 2.0]
    assert td.target(0.0, 1, False) == pytest.approx(1.8)
    td.update(0, 0.0, 1, False)
    assert td.V[0] == pytest.approx(1.4)
    mc = m.MonteCarloPrediction(2, gamma=0.9, alpha=0.5)
    mc.V[:] = [1.0, 2.0]
    mc.update(0, mc.target([0.0, 4.0], 0))
    assert mc.V[0] == pytest.approx(2.3)
    ex = m.TD0Prediction(2, gamma=0.8, alpha=0.1)       # exercise 1
    ex.V[:] = [2.0, 5.0]
    assert ex.target(1.0, 1, False) == pytest.approx(5.0)
    assert ex.update(0, 1.0, 1, False) == pytest.approx(3.0) and ex.V[0] == pytest.approx(2.3)
    assert ex.target(1.0, 1, True) == pytest.approx(1.0)  # exercise 2: terminal


def test_lesson04_mc_sample_mean_and_td_converge(lesson):
    m = lesson(f"{CH}/lesson_04_mc_td_prediction", "prediction")
    env = RandomWalk(n=5, seed=1)
    pi = TabularPolicy(env.uniform_policy())
    rng = np.random.default_rng(1)
    mc = m.MonteCarloPrediction(env.n_states, terminal=env.terminal)  # alpha = 1/N -> sample mean
    td = m.TD0Prediction(env.n_states, alpha=0.02, init_value=0.5, terminal=env.terminal)
    for _ in range(3000):
        traj = rollout(env, pi, rng)
        mc.learn_episode(traj)
        td.learn_episode(traj)
    assert np.max(np.abs(mc.V[1:-1] - env.true_values())) < 0.05
    assert np.max(np.abs(td.V[1:-1] - env.true_values())) < 0.05


@pytest.mark.parametrize("pi_L, sampled, expected", [
    (0.75, "R", {"Sarsa": (1.0, 1.5), "Expected Sarsa": (3.7, 2.85), "Q-learning": (4.6, 3.3)}),
    (0.5, "L", {"Sarsa": (4.6, 3.3), "Expected Sarsa": (2.8, 2.4), "Q-learning": (4.6, 3.3)}),
])
def test_lesson05_targets(lesson, pi_L, sampled, expected):
    m = lesson(f"{CH}/lesson_05_sarsa_qlearning", "td_control")
    assert m.worked_example(pi_L, sampled) == {k: pytest.approx(v) for k, v in expected.items()}


def test_lesson05_q_learning_finds_cliff_edge_path(lesson):
    m = lesson(f"{CH}/lesson_05_sarsa_qlearning", "td_control")
    env = cliff_walking(seed=0)
    agent = m.QLearning(env.n_states, env.n_actions, alpha=0.5, epsilon=0.1, seed=0)
    agent.train(env, 300)
    path = env.greedy_path(TabularPolicy.greedy(agent.Q).table)
    assert len(path) - 1 == 13  # the optimal path has 13 steps (return -13)


def test_lesson05_glie_schedule(lesson):
    m = lesson(f"{CH}/lesson_05_sarsa_qlearning", "td_control")
    assert m.EpsilonSchedule.constant(0.1)(10_000) == 0.1
    assert m.EpsilonSchedule.glie()(10_000) == pytest.approx(1e-4)


def test_lesson05_maximization_bias(lesson):
    m = lesson(f"{CH}/lesson_05_sarsa_qlearning", "maximization_bias")
    single, double = m.single_and_double_estimates(10, 5, 5000)
    assert single.mean() > 0.3 and abs(double.mean()) < 0.05


def test_lesson06_examples(lesson):
    m = lesson(f"{CH}/lesson_06_multistep_traces", "multistep")
    V = np.array([0.0, 0.0, 3.0])
    assert m.n_step_return([1, 2, 0], [0, 1, 2, 2], 0, 2, 0.9, V) == pytest.approx(5.23)
    assert m.n_step_return([1, 2], [0, 1, 2], 0, 2, 0.9, V) == pytest.approx(2.8)   # exercise 1
    tr = m.AccumulatingTrace(3, gamma=1.0, lam=0.8)
    assert [tr.visit(s)[0] for s in (0, 1, 2)] == pytest.approx([1.0, 0.8, 0.64])
    assert tr.visit(0)[0] == pytest.approx(1.512)                                       # exercise 2


def test_lesson06_forward_equals_backward_offline(lesson):
    m = lesson(f"{CH}/lesson_06_multistep_traces", "multistep")
    env = RandomWalk(n=7, left_reward=-1, seed=3)
    pi = TabularPolicy(env.uniform_policy())
    rng = np.random.default_rng(3)
    V = rng.normal(0, 0.3, env.n_states)
    V[env.terminal] = 0
    for _ in range(5):
        traj = rollout(env, pi, rng)
        for lam in (0.0, 0.5, 0.9, 1.0):
            np.testing.assert_allclose(m.offline_lambda_return_update(V, traj, 0.1, 1.0, lam),
                                       m.offline_td_lambda_update(V, traj, 0.1, 1.0, lam), atol=1e-10)


def test_lesson06_lambda_zero_is_td0(lesson):
    m = lesson(f"{CH}/lesson_06_multistep_traces", "multistep")
    td0 = lesson(f"{CH}/lesson_04_mc_td_prediction", "prediction")
    env = RandomWalk(n=5, seed=0)
    traj = rollout(env, TabularPolicy(env.uniform_policy()), np.random.default_rng(0))
    a = m.TDLambdaPrediction(env.n_states, lam=0.0, alpha=0.1, init_value=0.5)
    b = td0.TD0Prediction(env.n_states, alpha=0.1, init_value=0.5, terminal=env.terminal)
    a.V[env.terminal] = 0
    a.learn_episode(traj)
    b.learn_episode(traj)
    np.testing.assert_allclose(a.V, b.V)


def test_lesson06_importance_sampling(lesson):
    m = lesson(f"{CH}/lesson_06_multistep_traces", "importance_sampling")
    assert m.importance_ratio(0.8, 0.4) == pytest.approx(2.0)
    with pytest.raises(m.SupportError):                                                 # exercise 3
        m.importance_ratio(0.2, 0.0)
    env = RandomWalk(n=5, seed=0)
    target = TabularPolicy(np.tile([0.25, 0.75], (env.n_states, 1)))
    behavior = TabularPolicy(env.uniform_policy())
    est = m.OffPolicyMCEvaluator(target, behavior)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        est.add(rollout(env, behavior, rng))
    true = env.evaluate_policy_exact(target.table)[3]
    assert abs(est.weighted - true) < 0.02 and abs(est.ordinary - true) < 0.1
