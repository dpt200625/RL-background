# RL-background

Study notes and runnable code for the foundations of Reinforcement Learning — **from David Silver's course to deep RL and modern research**.

The lecture notes ("RL Study Notes", v2.2) are rewritten here in English without changing their content; each lesson pairs the text with small, CPU-only code that reproduces the formulas, the worked examples and the exercises.

## Purpose of the notes

Understand the formulation, formulas, update mechanisms and pipeline of RL in order to read research independently. The notes stop at analyzing the data, the target, the learning module and the evaluation protocol; they do not require writing code. They are a selective interpretation and do not replace the four original sources. *This repository adds the code side: every lesson re-computes its examples and runs a small experiment.*

**Knowledge axis.** David Silver, Lectures 1–9, sets the order and the main intuition. Sutton–Barto, Chapters 2–13, adds definitions, conditions and the steps of derivations. Berkeley CS185/285 **Spring 2026** extends to deep RL, imitation, model-based/world models, offline RL, inference, multi-task/hierarchy and LLM RL. Boyd–Vandenberghe sits in the final supporting part, keeping only the optimization needed to read methods.

**How to read.** Read Sections 1–6 first; then Sections 7–11 to connect with Silver 6–9. Sections 12–18 are the Berkeley extension. Section 19 on optimization can be read when meeting a gradient, a KL constraint or a Lagrangian. Every section uses the same seven-part frame. After a formula comes how to read it, the meaning of each component or a guiding question; the example in the next part helps carry out the computation. Earlier knowledge is linked so it can be re-read when needed.

**Level and sources.** The notes do not collect every algorithm into one loss. The objective describes the problem; the update/loss describes how to learn; the metric measures the output. Conditions of tabular results are stated separately and are not generalized to neural networks. Formulas are checked against the sources in Appendix B; notation and conditions of use are explained in the lessons; the numeric examples and exercises are **self-constructed illustrations**, not experimental results of the sources.

| Source | Role and parts used |
|---|---|
| Silver, UCL 2015 | Lectures 1–5: MDP, Bellman, DP, MC/TD, control; 6–9: approximation, policy gradient, planning, exploration. |
| Sutton–Barto, 2e | Ch. 2–8: bandits and tabular methods; Ch. 9–13: approximation, off-policy, traces and policy gradient. Author PDF, 2020 revision. |
| Berkeley, Spring 2026 | Lectures 2–3, 4–20, 23–25; content chosen to directly supplement the foundation axis. No homework/API summary. |
| Boyd–Vandenberghe | Ch. 2–3; §§4.1–4.2, 4.7; Ch. 5; §§9.1–9.3. The solver families are not implemented in full. |

## Contents

| Part | Folder | Sections | Status |
|---|---|---|---|
| — | [NOTATION.md](NOTATION.md) | Notation used throughout | ready |
| I. RL foundations and tabular methods | [01_foundations_tabular](01_foundations_tabular/README.md) | 1–6 | **ready**: 6 lessons, notebooks + code + experiments |
| II. Function approximation and policy optimization | [02_function_approximation_policy_optimization](02_function_approximation_policy_optimization/README.md) | 7–9 | planned |
| III. Exploration, planning and world models | [03_exploration_planning_world_models](03_exploration_planning_world_models/README.md) | 10–12 | planned |
| IV. Learning from data and inference | [04_learning_from_data_inference](04_learning_from_data_inference/README.md) | 13–15 | planned |
| V. Generalization and modern research | [05_generalization_modern_research](05_generalization_modern_research/README.md) | 16–18 | planned |
| VI. Supporting optimization background | [06_optimization_background](06_optimization_background/README.md) | 19 | planned |
| VII. Practice and reference | [07_practice_reference](07_practice_reference/README.md) | Appendix A (answers), Appendix B (source map), references | ready |

## Repository layout

```
RL-background/
├── README.md, NOTATION.md
├── requirements.txt          # Chapter I: numpy, matplotlib, jupyter, pytest (pinned)
├── requirements-deep.txt     # Chapter II+: torch (CPU), gymnasium
├── pyproject.toml            # installs the shared package `rlcore` (pip install -e .)
├── rlcore/                   # shared building blocks used by many lessons
│   ├── mdp.py                #   TabularMDP: P^a_ss', R^a_s, sampling, exact policy evaluation
│   ├── envs.py               #   GridWorld, cliff_walking, RandomWalk
│   ├── policies.py           #   TabularPolicy, EpsilonGreedyPolicy
│   ├── rollout.py            #   Trajectory, rollout, evaluate_policy
│   ├── plotting.py, artifacts.py
├── 01_foundations_tabular/   # one folder per chapter: README.md + one folder per lesson
│   └── lesson_XX_<topic>/
│       ├── lesson.ipynb      #   the lecture (markdown) interleaved with code cells
│       ├── <component>.py    #   the algorithms taught in the lesson, importable and runnable
│       └── experiment_*.py   #   a small CPU experiment, writes to artifacts/
├── artifacts/                # run outputs (figures, JSON), git-ignored
├── tests/                    # pytest: every worked example and exercise answer
└── tools/run_notebooks.py    # execute notebooks in place
```

Design rules for the code:

* **The lesson code is the algorithm.** Each lesson folder implements what that lesson teaches, one class per concept (`PolicyEvaluation`, `TD0Prediction`, `Sarsa`, ...). `rlcore` only holds what several lessons reuse.
* **Equations are traceable.** Docstrings and comments cite the equation numbers of the notes, e.g. `Eq. (13)`.
* **Small and CPU-only.** Every experiment has a command-line budget (`--runs`, `--episodes`) and runs in seconds with the defaults.
* **Results are reproducible.** Seeds are explicit; outputs go to `artifacts/<chapter>/<lesson>/` as PNG + JSON.

## Setup

Python 3.10+ (tested with 3.12, Windows, CPU only). A virtual environment outside the OneDrive folder keeps OneDrive from syncing thousands of files:

```bash
python -m venv ~/.venvs/rl-background
source ~/.venvs/rl-background/Scripts/activate   # Linux/macOS: ~/.venvs/rl-background/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name rl-background --display-name "Python (rl-background)"
```

With conda instead: `conda create -n rl-background python=3.12`, `conda activate rl-background`, then the same `pip` lines.

## Usage

Open a lesson notebook (kernel *Python (rl-background)*) and read top to bottom, or run the code directly:

```bash
cd 01_foundations_tabular/lesson_05_sarsa_qlearning
python td_control.py                         # recomputes the worked example of Section 5
python experiment_cliff_walking.py --runs 20  # figure + JSON in artifacts/
```

Checks:

```bash
pytest                                        # every worked example and exercise answer
python tools/run_notebooks.py 01_foundations_tabular   # re-execute the notebooks
```

## References

1. David Silver. *Reinforcement Learning*, UCL, 2015. Lectures 1–9. https://www.davidsilver.uk/teaching/
2. Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018; author PDF, 2020 revision. http://incompleteideas.net/book/RLbook2020.pdf
3. Sergey Levine and the CS185/285 team, UC Berkeley. *Deep Reinforcement Learning*, Spring 2026. https://rail.eecs.berkeley.edu/deeprlcourse/
4. Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*, Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/

Related reference implementations: the experiments follow Sutton–Barto's examples (random walk, cliff walking, n-step/TD(λ) curves), also reproduced in [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction); the model format `outcomes[s][a] = [(prob, next_state, reward)]` follows gymnasium's toy-text environments, as in [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) (exercises for Silver's course and Sutton–Barto).
