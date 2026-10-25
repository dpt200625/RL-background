# I. RL foundations and tabular methods

Sections 1–6 of the notes. Everything here is tabular: states and actions are few enough to keep one number per state or state–action pair, so every formula can be computed exactly and checked. Foundations: Silver Lectures 1–5; Sutton–Barto Ch. 1, 3–7, 12, §5.5.

| # | Lesson | Learning goal | Code | Experiment (CPU, default budget) |
|---|---|---|---|---|
| 1 | [RL, MDP and return](lesson_01_rl_mdp_return/lesson.ipynb) | Describe a task by state, action, reward and horizon; distinguish observation from state; compute the return and identify the objective. | `returns.py`, `interaction.py`, `observability.py` | in the notebook |
| 2 | [Value functions and the Bellman equations](lesson_02_value_bellman/lesson.ipynb) | Compute $v_\pi, q_\pi$ and explain why Bellman links present value with the future; distinguish evaluation from optimality. | `bellman.py` | in the notebook |
| 3 | [Dynamic programming and GPI](lesson_03_dynamic_programming/lesson.ipynb) | Perform a Bellman backup; distinguish policy evaluation, policy improvement and value iteration. | `dynamic_programming.py` | `experiment_gridworld_dp.py` (~3 s) |
| 4 | [Monte Carlo and TD prediction](lesson_04_mc_td_prediction/lesson.ipynb) | Distinguish a sample return from a bootstrap target; compute one MC/TD update and explain bias, variance. | `prediction.py` | `experiment_random_walk.py` (~15 s) |
| 5 | [Sarsa, Expected Sarsa and Q-learning](lesson_05_sarsa_qlearning/lesson.ipynb) | Compute the three targets and recognize the target/behavior policy; state the scope of the tabular Q-learning guarantee. | `td_control.py`, `maximization_bias.py` | `experiment_cliff_walking.py` (~13 s) |
| 6 | [Multi-step returns, traces and off-policy estimation](lesson_06_multistep_traces/lesson.ipynb) | Explain credit assignment over several steps; compute an n-step target and an importance ratio; understand the scope of the forward/backward equivalence. | `multistep.py`, `importance_sampling.py` | `experiment_nstep_lambda.py` (~9 s), `experiment_off_policy_is.py` (~8 s) |

## How the lessons connect

```
L1  return G_t, objective J ─┐
L2  Bellman: v_pi, q_pi, q_* ─┼─> L3  DP: known model, expectation + bootstrap
                              │        │ replace the model by experience
                              │        v
                              └─> L4  MC (sample, no bootstrap) / TD (sample + bootstrap)   prediction
                                       │ use the value to change the policy
                                       v
                                  L5  Sarsa / Expected Sarsa / Q-learning                    control
                                       │ credit over several steps, data from another policy
                                       v
                                  L6  n-step, eligibility traces, importance sampling
```

## Equation index

| Eq. | Formula | Where in the code |
|---|---|---|
| (1) | $G_t = \sum_k \gamma^k R_{t+k+1}$ | `lesson_01/returns.py::discounted_return` |
| (2) | $J(\theta) = \mathbb E_{\pi_\theta}[G_0]$ | `lesson_01/returns.py::ObjectiveEstimator`, `rlcore.evaluate_policy` |
| (3) | $G_t = R_{t+1} + \gamma G_{t+1}$ | `lesson_01/returns.py::returns_to_go` |
| (4) | $v_\pi, q_\pi$ as expected returns | `lesson_02` notebook (rollout averages vs `TabularMDP.evaluate_policy_exact`) |
| (5) | Bellman expectation for $v_\pi$ | `lesson_02/bellman.py::BellmanEquations.expectation_backup`, `sampled_target` |
| (6) | $q_\pi = \mathcal R + \gamma \sum \mathcal P v_\pi$ | `BellmanEquations.q_from_v`, `rlcore.TabularMDP.lookahead` |
| (7) | $v_\pi = \sum_a \pi q_\pi$ | `BellmanEquations.v_from_q` |
| (8) | Bellman optimality for $q_*$ | `BellmanEquations.optimality_backup`, `q_star` |
| (9) | Policy evaluation sweep | `lesson_03/dynamic_programming.py::BackupSweep`, `PolicyEvaluation` |
| (10) | Greedy improvement | `PolicyImprovement`, `rlcore.TabularPolicy.greedy` |
| (11) | Value iteration sweep | `ValueIteration` |
| (12) | MC update | `lesson_04/prediction.py::MonteCarloPrediction` |
| (13)–(14) | TD error, TD(0) update | `TD0Prediction.td_error`, `TD0Prediction.update` |
| (15) | Sarsa | `lesson_05/td_control.py::Sarsa.target` |
| (16) | Expected Sarsa | `ExpectedSarsa.target` |
| (17) | Q-learning | `QLearning.target` (and `DoubleQLearning`) |
| (18) | n-step return | `lesson_06/multistep.py::n_step_return`, `NStepTDPrediction` |
| (19) | Accumulating trace | `AccumulatingTrace` |
| (20) | TD(λ) update | `TDLambdaPrediction`, `offline_td_lambda_update` (vs `offline_lambda_return_update`) |
| (21) | Importance ratio | `lesson_06/importance_sampling.py::importance_ratio`, `OffPolicyMCEvaluator` |

## Run

```bash
cd 01_foundations_tabular/lesson_04_mc_td_prediction
python prediction.py                     # worked example of the section
python experiment_random_walk.py --runs 100 --episodes 100
```

Every worked example and exercise answer of this chapter is also checked by `pytest tests/test_chapter01.py`.
