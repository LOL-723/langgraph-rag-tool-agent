import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_VENV = PROJECT_ROOT / ".venv"
EXPECTED_PYTHON = PROJECT_VENV / "Scripts" / "python.exe"


def ensure_project_runtime() -> None:
    if not EXPECTED_PYTHON.exists():
        raise RuntimeError(
            f"Project virtual environment is missing: {PROJECT_VENV}. "
            "Create it and install dependencies before starting FastAPI."
        )

    current_python = Path(sys.executable).resolve()
    expected_python = EXPECTED_PYTHON.resolve()
    if current_python != expected_python:
        raise RuntimeError(
            "FastAPI is running with a different Python environment. "
            f"Current: {current_python}. Expected: {expected_python}. "
            "Start the app with scripts/dev.ps1 or run "
            r".\.venv\Scripts\python.exe -m uvicorn main:app --reload"
        )

    os.environ.setdefault("VIRTUAL_ENV", str(PROJECT_VENV))
    scripts_dir = str(PROJECT_VENV / "Scripts")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if scripts_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([scripts_dir, *path_parts])
