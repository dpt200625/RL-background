"""Execute lesson notebooks in place (each one runs from its own folder), CPU only.

Usage:
    python tools/run_notebooks.py                      # every notebook in the repo
    python tools/run_notebooks.py 01_foundations_tabular
    python tools/run_notebooks.py path/to/lesson.ipynb
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore", message="Proactor event loop does not implement add_reader")  # pyzmq on Windows


def find_notebooks(targets: list[str]) -> list[Path]:
    if not targets:
        targets = [str(REPO_ROOT)]
    found = []
    for t in targets:
        p = Path(t).resolve()
        if p.suffix == ".ipynb":
            found.append(p)
        else:
            found += [nb for nb in sorted(p.rglob("*.ipynb")) if ".ipynb_checkpoints" not in nb.parts]
    return found


def run(path: Path, timeout: int = 900) -> float:
    nb = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    return time.perf_counter() - start


def main() -> int:
    failed = []
    for path in find_notebooks(sys.argv[1:]):
        rel = path.relative_to(REPO_ROOT)
        try:
            print(f"ok   {rel} ({run(path):.0f}s)", flush=True)
        except Exception as err:  # report every failing notebook, not only the first
            failed.append(rel)
            print(f"FAIL {rel}: {type(err).__name__}: {str(err).splitlines()[-1] if str(err) else ''}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
