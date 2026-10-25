# IV. Learning from data and inference

Sections 13–15 of the notes. **Status: planned** — same layout as Chapter I.

| # | Section | Foundations | Learning goal | Planned code (CPU-light) |
|---|---|---|---|---|
| 13 | Imitation learning and inverse RL | Extension: Berkeley 2–3; Berkeley 13 (IRL and adversarial imitation). | Distinguish learning actions from demonstrations from learning reward from behavior; understand covariate shift and reward ambiguity. | behavioral cloning vs DAgger on a grid with an expert; tabular maximum-entropy IRL |
| 14 | Offline RL, data support and offline-to-online | Extension: Berkeley 17–18. Related Sutton–Barto Ch. 11. | Distinguish offline from off-policy; explain OOD action values, CQL/IQL and the limits of offline evaluation. | offline Q-learning on a fixed dataset with limited support; tabular CQL; trajectory importance sampling for OPE |
| 15 | Variational inference and control as inference | Extension: Berkeley 11–13. Probability and optimization background: Boyd Ch. 3, 5. | Understand posterior, ELBO and KL; distinguish the likelihood objective of a model from the return objective of a controller. | ELBO / KL on a small latent-variable model; soft value iteration |

Planned folders: `lesson_13_imitation_inverse_rl/`, `lesson_14_offline_rl/`, `lesson_15_variational_inference_control/`.
