# II. Function approximation and policy optimization

Sections 7–9 of the notes. **Status: planned** — the folder layout follows Chapter I (one `lesson_XX_<topic>/` per section with `lesson.ipynb` + component `.py` files + a small CPU experiment). Extra packages: `pip install -r requirements-deep.txt` (CPU PyTorch, gymnasium).

| # | Section | Foundations | Learning goal | Planned code (CPU-light) |
|---|---|---|---|---|
| 7 | Function approximation and Deep Q-Networks | Silver 6; Sutton–Barto Ch. 9–11. Extension: Berkeley 4, 7–8. | Distinguish the true value, a table estimate and a neural estimate; understand semi-gradient, replay, target network and the deadly triad. | linear semi-gradient TD with features; DQN with replay buffer and target network, Double DQN (CartPole) |
| 8 | Policy gradient, REINFORCE and actor–critic | Silver 7; Sutton–Barto Ch. 13. Extension: Berkeley 5–6. | Derive the meaning of the score-function gradient; distinguish return, baseline and advantage; understand actor/critic and GAE. | REINFORCE with baseline, one-step actor–critic, GAE (CartPole) |
| 9 | Modern policy updates: TRPO, PPO and maximum-entropy RL | Extension: Berkeley 9–10, 12–13; related Silver 7 and Boyd Ch. 5. | Understand why a large policy update makes old data less suitable; distinguish the PPO surrogate from the return objective; explain entropy and the SAC mechanism. | PPO-clip surrogate and entropy bonus (CartPole); small SAC (Pendulum) |

Planned folders: `lesson_07_function_approximation_dqn/`, `lesson_08_policy_gradient_actor_critic/`, `lesson_09_trpo_ppo_max_entropy/`.
