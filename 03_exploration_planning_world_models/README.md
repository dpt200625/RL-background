# III. Exploration, planning and world models

Sections 10–12 of the notes. **Status: planned** — same layout as Chapter I.

| # | Section | Foundations | Learning goal | Planned code (CPU-light) |
|---|---|---|---|---|
| 10 | Exploration, bandits and information gathering | Silver 9; Sutton–Barto Ch. 2. Extension: Berkeley 19, 23. | Distinguish exploration from stochastic actions; understand epsilon-greedy, UCB, Thompson sampling and the limits of intrinsic reward. | 10-armed testbed: epsilon-greedy, optimistic initialization, UCB, Thompson sampling, gradient bandit |
| 11 | Learning, planning, Dyna and model-based RL | Silver 8; Sutton–Barto Ch. 8. Extension: Berkeley 15–16. | Distinguish real/simulated experience; understand Dyna, search and MPC; recognize model exploitation. | Dyna-Q on a maze (real vs simulated updates); random-shooting MPC with a learned model |
| 12 | World models, latent states and partial observability | Extension: Berkeley 16 (latent-state models), Berkeley 14 (POMDP). Related Silver 1 and 8. | Describe a world model at the level of representation/dynamics/reward; distinguish planning in the model from learning an actor in imagination. | partially observed task: single frame vs history; a small latent dynamics + reward model |

Planned folders: `lesson_10_exploration_bandits/`, `lesson_11_planning_dyna_mbrl/`, `lesson_12_world_models_pomdp/`.
