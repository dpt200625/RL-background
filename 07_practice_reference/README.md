# VII. Practice and reference

The lesson notebooks of Chapter I already contain their exercises with collapsible answers and code that checks them. This page gathers **all** exercises and answers (Appendix A) and the source map (Appendix B).

## Appendix A. Answers to the exercises

> **How to check yourself.** Do the questions before reading the answers. The numbers below are results of self-constructed examples; a solution needs both the right number and the right reason. If you only remember the formula but do not identify the terminal, the policy or the data support, the learning goal is not reached yet.

<details><summary><b>1. RL, MDP and return</b></summary>

1. An episode has rewards $2, -1, 3$, $\gamma = 0.5$: compute $G_0, G_1$.
2. Why might an image of the current position not be enough for a motion-control problem?
3. If the simulator stops because of a run limit, is that necessarily a terminal of the task?

**Answers.** (1) $G_0 = 2 - 0.5 + 0.25\cdot 3 = 2.25$, $G_1 = -1 + 0.5\cdot 3 = 0.5$. (2) The same position can have a different velocity or hidden state; the consequences of an action differ even though the present looks the same. (3) No: a computational cutoff may just be a truncation. If the horizon limit is part of the task, the remaining time may also need to be included in the state.
</details>

<details><summary><b>2. Value and Bellman</b></summary>

1. If in the example $v_\pi(u) = 2$, what are $q_\pi(s,a)$ and $v_\pi(s)$?
2. Why does being greedy with respect to $q_\pi$ change the policy and therefore the value?
3. With only $v_\pi$ but no model: can we always choose the greedy action?

**Answers.** (1) $q_\pi(s,a) = 2 + 0.9\cdot 0.75\cdot 2 = 3.35$; the uniform policy has $v_\pi(s) = (3.35 + 3)/2 = 3.175$. (2) Greedy changes the action probabilities, so the future trajectories and the value also change. (3) Not in general; a model/reward is needed to look ahead, or an appropriate action-value/actor.
</details>

<details><summary><b>3. Dynamic programming</b></summary>

1. In the example, change $\gamma = 0.4$: does the final greedy policy choose $a$ or $b$?
2. If we update $u$ first and then use the new value for $s$ immediately within one sweep, does the result of round 1 differ from a synchronous backup?
3. Does contraction with a wrong model guarantee the correct policy in the real environment?

**Answers.** (1) Action $a$ gives $0.4\cdot 4 = 1.6$, $b$ gives $2$, so choose $b$. (2) Yes: an in-place sweep updates $u = 4$ then uses it immediately for $s$, giving $3.6$ within one sweep; the synchronous backup uses $V_0(u) = 0$. (3) No: the contraction of a wrong model only solves the fixed point of that model.
</details>

<details><summary><b>4. MC and TD</b></summary>

1. With $V(s) = 2, V(u) = 5, R_{t+1} = 1, \gamma = 0.8, \alpha = 0.1$, compute the TD target, $\delta_t$ and the new estimate.
2. If the next state is terminal, what is the target?
3. Why can the MC target and the TD target differ on the same transition?

**Answers.** (1) Target $1 + 0.8\cdot 5 = 5$; $\delta = 5 - 2 = 3$; new $V(s)$ $2.3$. (2) The target is only $R_{t+1}$ because the terminal value is zero. (3) MC uses the realized rewards of the remainder; TD uses the learned next-state estimate, so they do not necessarily coincide.
</details>

<details><summary><b>5. Sarsa and Q-learning</b></summary>

1. Keep the example but $\pi(L \mid u) = 0.5$, $A_{t+1} = L$. Compute the three targets and the three new estimates.
2. Is a fixed $\epsilon$-greedy $0.1$ GLIE?
3. An action never seen in the dataset but with a large $Q$: does the tabular theorem guarantee that this estimate is correct?

**Answers.** (1) Sarsa target $4.6$, estimate $3.3$; Expected Sarsa target $2.8$, estimate $2.4$; Q-learning target $4.6$, estimate $3.3$. (2) No: a fixed $\epsilon$ is not greedy in the limit; infinite visitation also needs its own proof. (3) No: the theorem requires coverage, it does not apply to a pair that has never been learned.
</details>

<details><summary><b>6. Multi-step, traces and off-policy</b></summary>

1. If the episode ends right after the two rewards of the example, what is the target?
2. Compute the trace at the next step if the state has just been revisited and the previous trace is $0.64$, $\gamma\lambda = 0.8$.
3. The target needs an action with probability $0.2$, the behavior probability is zero: can importance sampling be used?

**Answers.** (1) No bootstrap: $1 + 0.9\cdot 2 = 2.8$. (2) $0.8\cdot 0.64 + 1 = 1.512$. (3) No: the support condition is violated, the ratio is undefined; there are no samples to evaluate that action.
</details>

<details><summary><b>7. Function approximation and DQN</b></summary>

1. A DQN has reward $2$, $\gamma = 0.9$, next target Q-values $(3, 5)$: compute $y$ when $d = 0$ and $d = 1$.
2. In the linear example, if $y = 1$, what is the new parameter?
3. Why not apply the tabular Q-learning theorem directly to DQN?

**Answers.** (1) $y = 2 + 0.9\cdot 5 = 6.5$ when nonterminal; $y = 2$ when terminal. (2) Prediction $2$, error $-1$, so $w = (-0.1, 0.8)^\top$. (3) Shared nonlinear parameters, the data distribution and bootstrap/off-policy make the assumptions and the update geometry different from tabular.
</details>

<details><summary><b>8. Policy gradient and actor–critic</b></summary>

1. Return $2$, baseline $3$: what is the sign of the policy update for the sampled action?
2. Two TD errors $1, 2$ with $\gamma\lambda = 0.8$: compute GAE at the first step.
3. Why not arbitrarily subtract an action-dependent baseline in REINFORCE?

**Answers.** (1) The advantage sample is $-1$, so the update goes against the log-gradient of the sampled action, decreasing its probability for a small local step. (2) $1 + 0.8\cdot 2 = 2.6$. (3) An action-dependent baseline can have a nonzero expected score contribution; using it needs an appropriate correction/derivation.
</details>

<details><summary><b>9. PPO and maximum-entropy RL</b></summary>

1. $\rho = 0.7$, $\epsilon = 0.2$, $\hat A = -2$: what are the two terms and the PPO minimum?
2. Does PPO clipping guarantee a small KL at every state?
3. Can changing the entropy temperature change the optimal policy?

**Answers.** (1) Unclipped term $-1.4$, clipped term $-1.6$, minimum $-1.6$. (2) No: the clipped surrogate does not enforce a pointwise/global KL constraint. (3) Yes: entropy changes the preference in the objective, so it can change the optimum.
</details>

<details><summary><b>10. Exploration</b></summary>

1. Why does epsilon-greedy retry even actions known for sure to be bad?
2. If the prediction error is high because of sensor noise that cannot be predicted, what can happen to curiosity?
3. How does a bandit "without state transitions" differ from navigating to a reward locked behind many actions?

**Answers.** (1) The random part does not take uncertainty into account, so it also tries actions known to be bad. (2) The policy can be attracted by unlearnable randomness instead of task progress. (3) A bandit tries an arm directly; navigation needs to reach the right state through a sequence of actions before seeing the reward.
</details>

<details><summary><b>11. Planning and model-based RL</b></summary>

1. In Dyna, does a simulated transition increase the number of real environment interactions?
2. Why can a small one-step prediction error still give a bad policy?
3. MPC optimizes an open-loop sequence but executes with replanning: which part is the feedback?

**Answers.** (1) It does not increase the real interaction count; it still increases model compute. (2) The training distribution differs from the planned trajectory distribution; model error at important state/actions or over a long rollout can be exploited by the optimizer. (3) The feedback lies in the real observation after the first action and in replanning.
</details>

<details><summary><b>12. World models</b></summary>

1. Is a low reconstruction loss enough to prove that reward prediction is correct?
2. When the actor is trained in imagination, does deployment necessarily have to search action sequences?
3. Why must an encoder that uses future observations during training be handled carefully at deployment?

**Answers.** (1) No: small details can decide the reward but contribute little to the reconstruction error. (2) No: the actor can choose actions directly from the latent estimate; planning is another choice. (3) Future observations do not exist at deployment; a corresponding causal estimator/filtering is needed and information leakage must be avoided.
</details>

<details><summary><b>13. Imitation and inverse RL</b></summary>

1. BC validation loss decreases but robot success decreases: is that a contradiction?
2. Can DAgger be used as is if the expert cannot be queried at new states?
3. Does expert behavior determine a unique reward?

**Answers.** (1) Not a contradiction: validation is on expert states, execution is on learner-induced states. (2) Not as is; the requirement of expert labels at the states the learner visits must be met, or the setting/method changed. (3) No: many reward hypotheses can explain the same demonstrations.
</details>

<details><summary><b>14. Offline RL</b></summary>

1. A dataset has one million transitions but only one action in each state: is it enough to evaluate every policy?
2. Is online Q-learning with replay offline RL?
3. Two importance ratios equal $2$ and $0.5$: what is the trajectory weight; does this trajectory prove that the estimator is stable?

**Answers.** (1) No: size does not replace action/state support. (2) No; if it still collects new interactions while learning, it is online off-policy. (3) Weight $1$; one realization does not describe the variance, coverage or bias of the estimator.
</details>

<details><summary><b>15. Inference</b></summary>

1. Log likelihood $-3$, posterior KL $1.2$: what is the ELBO?
2. Why is KL not a symmetric distance?
3. Is a derivation that allows changing the stochastic transition probabilities to get a high reward the original control problem?

**Answers.** (1) ELBO $= -3 - 1.2 = -4.2$. (2) Swapping the two arguments changes the distribution of the expectation and the log ratio; the values are in general different. (3) No, unless the right to control the dynamics is part of the real problem; naive inference can solve a different, optimistic problem.
</details>

<details><summary><b>16. Multi-task and hierarchy</b></summary>

1. $\psi = (3, 2)^\top$, $u = (1, -1)^\top$: compute $q_\pi$.
2. Does changing the reward weights guarantee that the old policy is optimal?
3. What goes wrong when relabeling the goal but keeping the reward unchanged?

**Answers.** (1) $q_\pi = 3 - 2 = 1$. (2) No: successor features evaluate the existing policy for the new reward; optimal control still needs improvement/coverage. (3) The transition–reward labels no longer match the task, which makes the critic target wrong.
</details>

<details><summary><b>17. LLM RL</b></summary>

1. If the token is the action, what must the state contain?
2. Do the reference model and the old policy have the same role?
3. All completions have the same score: what are the centered group rewards; what does it say about the learning signal in the batch?

**Answers.** (1) The prompt and the generated prefix, plus the necessary information/task state. (2) No: the old policy is the sampling policy for the ratio; the reference is a distribution used to regularize. (3) The centered rewards are zero, the batch has no relative outcome signal under that rule; regularizers or other signals can still create an update.
</details>

<details><summary><b>18. Theory and evaluation</b></summary>

1. $\gamma = 0.8$, two value vectors $5$ apart: what is the bound after an exact backup?
2. Does a theorem that needs infinite visitation prove that a finite dataset is enough?
3. Why can a comparison only in environment steps ignore the cost of model-based methods?

**Answers.** (1) Max-norm bound $0.8\cdot 5 = 4$. (2) No: asymptotic visitation does not quantify finite-sample sufficiency. (3) Model rollouts, gradient steps and search have their own cost; interaction efficiency is not total computational efficiency.
</details>

<details><summary><b>19. Convex Optimization</b></summary>

1. With the example but the constraint $x \ge 1$, what are the optimum and the multiplier?
2. Does a convex squared error in linear weights make the return objective of a neural policy convex?
3. Does a weighted sum always obtain every Pareto solution in a nonconvex set?

**Answers.** (1) $x^* = 2$, the constraint is inactive and $\lambda_1^* = 0$. (2) No: that is a different objective, a different variable, and the distribution may depend on the policy. (3) No: unsupported Pareto points may not be the optimum of any positive weighted sum.
</details>

## Appendix B. Source map for checking and further study

**How to trace the sources.** The formulas of the main part come with explanations of how to read and use them. To check a definition or assumptions, open the corresponding lecture/chapter in the table below; common notation is defined at the start of the notes ([NOTATION.md](../NOTATION.md)). The Berkeley links point to the slides of the Spring 2026 course.

| Section | Keywords | Sources to open |
|---|---|---|
| 1 | RL, MDP, return | Silver 1–2; Sutton–Barto §§3.1–3.4 |
| 2 | Value, Bellman | Silver 2; Sutton–Barto §§3.5–3.7 |
| 3 | DP, GPI | Silver 3; Sutton–Barto §§4.1–4.6 |
| 4 | MC, TD prediction | Silver 4; Sutton–Barto §§5.1–5.2, 6.1–6.3 |
| 5 | Sarsa, Q-learning | Silver 5; Sutton–Barto §§6.4–6.7 |
| 6 | n-step, traces, IS | Silver 4–5; Sutton–Barto Ch. 7, §§12.1–12.5, §5.5 |
| 7 | Approximation, DQN | Silver 6; Sutton–Barto Ch. 9–11; Berkeley 2026, Lecture 7: Value-Based RL; Lecture 8: Q-learning in Practice |
| 8 | PG, actor–critic | Silver 7; Sutton–Barto Ch. 13; Berkeley 2026, Lecture 5: Policy Gradients; Lecture 6: Actor Critic |
| 9 | PPO, TRPO, SAC | Berkeley 2026, Lecture 9: Advanced PG Part 1; Lecture 10: Advanced PG Part 2; Lecture 13: Control as Variational Inference |
| 10 | Exploration | Silver 9; Sutton–Barto Ch. 2; Berkeley 2026, Lecture 19: Exploration; Lecture 23: Advanced Exploration |
| 11 | Dyna, model-based, MPC | Silver 8; Sutton–Barto Ch. 8; Berkeley 2026, Lecture 15: Model-Based RL Part 1; Lecture 16: Model-Based RL Part 2 |
| 12 | World models, POMDP | Berkeley 2026, Lecture 16: Latent-state models; Lecture 14: Sequences, LLMs and partial observability |
| 13 | BC, DAgger, IRL | Berkeley 2026, Lecture 2: Behavioral Cloning; Lecture 3: BC Part 2; Lecture 13: IRL and adversarial imitation |
| 14 | Offline RL | Berkeley 2026, Lecture 17: Offline RL Part 1; Lecture 18: Offline RL Algorithms |
| 15 | VI, control as inference | Berkeley 2026, Lecture 11: Variational Inference; Lecture 12: VI in RL; Lecture 13: Control as VI |
| 16 | Multi-task, goals, options | Berkeley 2026, Lecture 24: Multi-Task and Hierarchical RL |
| 17 | LLM RL, preferences | Berkeley 2026, Lecture 14: RL with Sequences and LLMs |
| 18 | Theory, evaluation | Berkeley 2026, Lecture 20: RL Theory; Lecture 25: Challenges and Open Problems |
| 19 | Convex Optimization | Boyd Ch. 2–3, §§4.1–4.2, 4.7, Ch. 5, §§9.1–9.3 |

**Bridge to research.** When reading a new paper, identify the setting, the training/deployment I/O, the objective, the estimator/update, the assumptions and the evidence first. Lead back to the corresponding foundation section: learned dynamics to Sections 11/12; offline methods to Section 14; preference/reward learning to Sections 13/17; transfer to Section 16. This is a rule for organizing notes, not for lumping problem families together.

**Scope not covered in depth.** To study average-reward and off-policy approximation in depth, read the whole of Ch. 10–11 and 13.6 of Sutton–Barto. Safe RL, MORL and reward-free RL need dedicated lectures/papers; these four sources give the background to keep reading, they do not fully establish those problem formulations.

## References

1. David Silver. *Reinforcement Learning*, UCL, 2015. Lectures 1–9. Author page and slide links: https://www.davidsilver.uk/teaching/. The slides were checked against the PDF version linked from the author page; accessed 11/10/2026.
2. Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018; author PDF, 2020 revision. Publisher page: https://mitpress.mit.edu/9780262352703/reinforcement-learning/. Author PDF: http://incompleteideas.net/book/RLbook2020.pdf.
3. Sergey Levine and the CS185/285 team, UC Berkeley. *Deep Reinforcement Learning*, Spring 2026. Lectures 2–20, 23–25 according to the map in Appendix B. https://rail.eecs.berkeley.edu/deeprlcourse/. The exact title in each PDF can be more specific than the label on the course page; content and formulas were checked against the PDFs.
4. Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*, Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/. Full PDF: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf. Sections used: Ch. 2–3, §§4.1–4.2, 4.7, Ch. 5, §§9.1–9.3.
