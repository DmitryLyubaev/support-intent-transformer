"""Canonical software checks. Never acquires inputs or runs an actual study."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="4",
)
os.environ["PYTHONPATH"] = (
    str(Path("tests/offline").resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
)
for command in (
    ["uv", "lock", "--check", "--offline"],
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "-m", "ruff", "format", "--check", "."],
    [sys.executable, "-m", "pytest", "-q"],
):
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
