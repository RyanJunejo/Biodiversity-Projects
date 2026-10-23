"""Configurable locations for datasets, caches and outputs.

Resolution order for each location: explicit argument, then environment
variable, then default.

    BIODIV_DATA_ROOT   root under which dataset paths in configs are resolved
    BIODIV_CACHE_DIR   Hugging Face / torch download cache
    BIODIV_OUTPUT_DIR  checkpoints, logs and result JSON files
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_root: Path
    cache_dir: Path
    output_dir: Path


def get_paths(
    data_root: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
) -> Paths:
    def pick(value, env_name, default):
        if value is not None:
            return Path(value).expanduser()
        return Path(os.environ.get(env_name, default)).expanduser()

    return Paths(
        data_root=pick(data_root, "BIODIV_DATA_ROOT", "data"),
        cache_dir=pick(cache_dir, "BIODIV_CACHE_DIR", "cache"),
        output_dir=pick(output_dir, "BIODIV_OUTPUT_DIR", "outputs"),
    )


def resolve(path: str | os.PathLike, base: Path) -> Path:
    """Return ``path`` unchanged if absolute, otherwise joined onto ``base``."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p
