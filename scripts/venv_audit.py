#!/usr/bin/env python3
"""Collect a machine-readable audit of the current Python environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "results" / "validation" / "venv_audit.json"

IMPORT_TARGETS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "sqlalchemy": "SQLAlchemy",
    "psycopg": "psycopg",
    "yaml": "PyYAML",
    "jsonschema": "jsonschema",
    "dotenv": "python-dotenv",
    "pydantic": "pydantic",
    "pytest": "pytest",
}


def _get_version(mod: Any, dist_name: str) -> str | None:
    for attr in ("__version__", "version"):
        if hasattr(mod, attr):
            try:
                value = getattr(mod, attr)
                return str(value)
            except Exception:
                pass
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version(dist_name)
    except Exception:
        return None


def _collect_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name, dist_name in IMPORT_TARGETS.items():
        try:
            mod = __import__(module_name)
        except Exception:
            versions[module_name] = None
            continue
        versions[module_name] = _get_version(mod, dist_name)
    return versions


def _run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _run_pytest() -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-m", "pytest"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    summary_line = None
    for line in reversed(proc.stdout.splitlines()):
        if line.strip():
            summary_line = line.strip()
            break
    if summary_line is None:
        for line in reversed(proc.stderr.splitlines()):
            if line.strip():
                summary_line = line.strip()
                break
    return {
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "summary_line": summary_line,
    }


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "python_version": sys.version,
        "packages": _collect_versions(),
        "pip_check": _run_command([sys.executable, "-m", "pip", "check"]),
        "pytest": _run_pytest(),
        "requirements_files": ["requirements.txt", "requirements-dev.txt"],
        "slug": "summit",
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote audit to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
