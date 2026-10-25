# RL-background

Reinforcement Learning foundations, from David Silver's course to deep RL and modern research: the lecture notes rewritten in English, with small runnable code for every formula, example and exercise.

**Goal:** understand the formulation, formulas, update mechanisms and pipeline of RL well enough to read research independently.

## 1. Setup (once)

```bash
python -m venv ~/.venvs/rl-background
source ~/.venvs/rl-background/Scripts/activate      # Linux/macOS: .../bin/activate
pip install -r requirements.txt && pip install -e .
python -m ipykernel install --user --name rl-background --display-name "Python (rl-background)"
```

CPU only, Python 3.10+. Chapter II onwards also needs `pip install -r requirements-deep.txt`.

## 2. Study path

Follow the reading order of the notes:

1. **Sections 1–6 first** (Chapter I): the tabular foundation everything else builds on.
2. **Then Sections 7–11** (Chapters II–III), connecting with Silver 6–9.
3. **Sections 12–18** (Chapters III–V) are the Berkeley extension.
4. **Section 19** (Chapter VI, optimization) whenever a gradient, KL constraint or Lagrangian appears.

Keep [NOTATION.md](NOTATION.md) open: $R_{t+1}$ is the reward of the transition $(S_t, A_t)$, $v_\pi$ / $q_\pi$ are true values, $V$ / $Q$ are estimates.

| Part | Sections | Topic | Status |
|---|---|---|---|
| [I](01_foundations_tabular/README.md) | 1–6 | RL foundations and tabular methods | ✅ ready |
| [II](02_function_approximation_policy_optimization/README.md) | 7–9 | Function approximation and policy optimization | planned |
| [III](03_exploration_planning_world_models/README.md) | 10–12 | Exploration, planning and world models | planned |
| [IV](04_learning_from_data_inference/README.md) | 13–15 | Learning from data and inference | planned |
| [V](05_generalization_modern_research/README.md) | 16–18 | Generalization and modern research | planned |
| [VI](06_optimization_background/README.md) | 19 | Supporting optimization background | planned |
| [VII](07_practice_reference/README.md) | A, B | All exercise answers, source map | ✅ ready |

## 3. How to study one lesson

Each lesson folder `lesson_XX_<topic>/` contains:

| File | Use it to |
|---|---|
| `lesson.ipynb` | read the lecture: 7 parts, each formula followed by a code cell |
| `<component>.py` | see the algorithm itself, one class per concept; `python <file>.py` recomputes the lecture's example |
| `experiment_*.py` | run a small experiment; figure + JSON go to `artifacts/<chapter>/<lesson>/` |

Every notebook follows the same 7-part frame of the notes:

1. **Quick intuition** – what the idea is for.
2. **The problem** – setting, input/output, objective vs metric.
3. **Core formulas** – each formula, how to read it, then code.
4. **Example and pipeline** – the worked numbers, recomputed; data → learning → use and evaluation.
5. **Comparison and limits** – what the method does *not* guarantee.
6. **Exercises** – try first, then check with code and the collapsible answers.
7. **Remember** – the key takeaway and what to read next.

Text in boxes that start with **Code** is added by this repo; everything else is the lecture.

## 4. Check your work

```bash
pytest                                                   # every worked example and exercise answer
python tools/run_notebooks.py 01_foundations_tabular     # re-run all notebooks of a chapter
```

## 5. Repository map

```
rlcore/          shared code: TabularMDP, GridWorld/RandomWalk, policies, rollouts, plotting
0X_<chapter>/    README.md + one folder per lesson
artifacts/       run outputs (git-ignored)
tests/, tools/   checks and notebook runner
```

## 6. Sources

| Source | Used for |
|---|---|
| [Silver, UCL 2015](https://www.davidsilver.uk/teaching/) | order and main intuition (Lectures 1–9) |
| [Sutton–Barto, 2e](http://incompleteideas.net/book/RLbook2020.pdf) | definitions, conditions, derivations (Ch. 2–13) |
| [Berkeley CS185/285, Spring 2026](https://rail.eecs.berkeley.edu/deeprlcourse/) | deep RL and research extensions (Lectures 2–20, 23–25) |
| [Boyd–Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) | optimization needed to read methods |

The numeric examples and exercises are self-constructed illustrations, not experimental results of the sources. The full source map per section is in [Part VII](07_practice_reference/README.md).
