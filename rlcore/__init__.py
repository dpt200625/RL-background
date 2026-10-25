"""rlcore: shared building blocks for the RL-background lessons.

Lesson folders keep the algorithm being taught in their own .py files;
rlcore only holds what several lessons reuse (MDPs, environments, policies,
rollouts, plotting and the artifacts folder).
"""
from .artifacts import ARTIFACTS_DIR, REPO_ROOT, artifact_path, save_figure, save_json
from .envs import GridWorld, RandomWalk, cliff_walking
from .mdp import Step, TabularMDP
from .policies import EpsilonGreedyPolicy, Policy, TabularPolicy, greedy_actions
from .rollout import Trajectory, evaluate_policy, rollout

__all__ = [
    "ARTIFACTS_DIR", "REPO_ROOT", "artifact_path", "save_figure", "save_json",
    "GridWorld", "RandomWalk", "cliff_walking",
    "Step", "TabularMDP",
    "EpsilonGreedyPolicy", "Policy", "TabularPolicy", "greedy_actions",
    "Trajectory", "evaluate_policy", "rollout",
]
