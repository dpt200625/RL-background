"""Plot helpers shared by the lessons (matplotlib only, no backend is forced)."""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .envs import ARROWS, GridWorld

# (dx, dy) of each action arrow in image coordinates: up, right, down, left
_ARROW_DXY = ((0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0))


def plot_grid_values(ax, env: GridWorld, v: np.ndarray, title: str = "", policy: np.ndarray | None = None,
                     annotate: bool = True, cmap: str = "viridis"):
    """Heatmap of state values on a GridWorld, optionally with greedy-action arrows."""
    grid = env.value_grid(v)
    im = ax.imshow(grid, cmap=cmap)
    for s in range(env.n_states):
        r, c = env.to_pos(s)
        ch = env.cell(s)
        if ch in "#":
            continue
        if ch in "GC":
            ax.text(c, r, ch, ha="center", va="center", color="white", fontweight="bold")
        elif policy is not None:
            a = int(np.asarray(policy)[s].argmax()) if np.ndim(policy) == 2 else int(policy[s])
            dx, dy = _ARROW_DXY[a]
            ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.15, color="white", length_includes_head=True)
        elif annotate:
            ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return im


def plot_curves(ax, curves: Mapping[str, np.ndarray], x: np.ndarray | None = None, xlabel: str = "",
                ylabel: str = "", title: str = "", band: bool = True):
    """Mean curve with a +-1 standard-error band. Each value is (runs, points) or (points,)."""
    for label, ys in curves.items():
        ys = np.atleast_2d(ys)
        mean = ys.mean(axis=0)
        xs = np.arange(len(mean)) if x is None else x
        line, = ax.plot(xs, mean, label=label)
        if band and ys.shape[0] > 1:
            se = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
            ax.fill_between(xs, mean - se, mean + se, color=line.get_color(), alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average along the last axis (valid part only)."""
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), -1, np.atleast_2d(x))


def new_figure(ncols: int = 1, nrows: int = 1, width: float = 5.0, height: float = 3.6):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    return fig, axes.ravel() if axes.size > 1 else axes[0, 0]


__all__ = ["ARROWS", "plot_grid_values", "plot_curves", "moving_average", "new_figure"]
