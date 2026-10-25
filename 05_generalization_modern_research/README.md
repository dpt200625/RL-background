# V. Generalization and modern research

Sections 16–18 of the notes. **Status: planned** — same layout as Chapter I.

| # | Section | Foundations | Learning goal | Planned code (CPU-light) |
|---|---|---|---|---|
| 16 | Multi-task, goal-conditioned and hierarchical RL | Extension: Berkeley 24. Related Silver 2, 8 and Sutton–Barto Ch. 3. | Distinguish task context, goal and preference; understand goal relabeling, successor features and options at the mechanism level. | goal-conditioned Q-learning with relabeling on a grid; successor features re-scored for new reward weights |
| 17 | RL with sequences and LLMs | Extension: Berkeley 14 (LLM RL, preference/verifier rewards). Related Section 8. | Map token generation to sequential decisions; distinguish SFT, RLHF, verifier reward and group baseline. | a toy token policy trained with a verifier reward: SFT vs policy gradient with a group-relative baseline |
| 18 | Theory, limits and how to read RL experiments | Sutton–Barto Ch. 3–4, 9–11. Extension: Berkeley 20, 25. | Identify the assumptions of a theorem; distinguish approximation, optimization and statistical error; read results with the right budget. | numerical check of the Bellman contraction bound; seed variability of a learning curve |

Planned folders: `lesson_16_multitask_goal_hierarchical/`, `lesson_17_sequences_llm_rl/`, `lesson_18_theory_evaluation/`.
