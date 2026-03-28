"""Index persistence: save/load LogMindIndex."""

from __future__ import annotations

import os
import pickle
from typing import Optional

from .models import LogMindIndex

DEFAULT_DIR = ".logmind"
DEFAULT_FILE = "index.pkl"


def _index_path(base_dir: str = ".") -> str:
    return os.path.join(base_dir, DEFAULT_DIR, DEFAULT_FILE)


def save_index(index: LogMindIndex, base_dir: str = ".") -> str:
    """Save index to disk."""
    dir_path = os.path.join(base_dir, DEFAULT_DIR)
    os.makedirs(dir_path, exist_ok=True)
    path = _index_path(base_dir)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_index(base_dir: str = ".") -> Optional[LogMindIndex]:
    """Load index from disk. Returns None if not found."""
    path = _index_path(base_dir)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def index_exists(base_dir: str = ".") -> bool:
    """Check if an index exists."""
    return os.path.exists(_index_path(base_dir))
