"""Load lesson modules by path: lesson folders are not Python packages (their names start with digits)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_lesson_module(lesson_dir: str, module: str):
    """Import <lesson_dir>/<module>.py; the lesson folder is put on sys.path for sibling imports."""
    folder = REPO_ROOT / lesson_dir
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    name = f"{folder.name}.{module}"
    spec = importlib.util.spec_from_file_location(name, folder / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lesson():
    return load_lesson_module
