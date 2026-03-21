"""
Centralized .env auto-loader for the project.

Usage:
    import env_loader  # load at import time, no-op if already loaded

This module uses python-dotenv's find_dotenv() to search upward from any file
in the project, finding the nearest .env file automatically.
It is idempotent — calling it multiple times is safe.
"""

import os
import threading
from pathlib import Path

_lock = threading.Lock()
_loaded = False


def load_env(override: bool = False) -> Path | None:
    """
    Find and load the .env file.

    Search order:
    1. project root (D:/Study/School/project_ky4)
    2. src/
    3. src/movierag/
    Walks UP from the caller's location to find the first .env file.

    Returns the path that was loaded, or None if nothing found.
    """
    global _loaded
    with _lock:
        if _loaded and not override:
            return None

        try:
            from dotenv import load_dotenv, find_dotenv
        except ImportError:
            # Fallback: manual walk-up search
            here = Path(__file__).resolve().parent
            for folder in [here, here.parent, here.parent.parent]:
                candidate = folder / ".env"
                if candidate.exists():
                    _load_manual(candidate)
                    _loaded = True
                    return candidate
            return None

        # find_dotenv walks upward from the current working directory
        # and then from the file location of THIS module
        env_path = find_dotenv(usecwd=True)
        if not env_path:
            # Try explicitly from the src/ directory where this file lives
            env_path = str(Path(__file__).resolve().parent / ".env")

        if env_path and Path(env_path).exists():
            load_dotenv(env_path, override=override)
            _loaded = True
            return Path(env_path)

        return None


def _load_manual(path: Path) -> None:
    """Fallback manual loader (no dotenv dependency)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)


# Auto-load when this module is imported
load_env()
