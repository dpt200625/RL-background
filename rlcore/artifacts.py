"""Where run outputs go: artifacts/<chapter>/<lesson>/<file>."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def artifact_path(*parts: str) -> Path:
    """Path inside artifacts/, creating parent folders. The last part is the file name."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, *parts: str, dpi: int = 120) -> Path:
    path = artifact_path(*parts)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(obj, *parts: str) -> Path:
    path = artifact_path(*parts)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path
