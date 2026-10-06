"""Where large data lives: $PROJECT_SCRATCH if set, else a fallback."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Seeds used by the checked-in production configs. Preparation must use the same ones.
DEFAULT_CONSTRUCTION_SEED = 1729
DEFAULT_PAIR_SEED = 2718


def scratch_root() -> Path:
    for name in ("PROJECT_SCRATCH", "GRIT_SCRATCH"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    return REPO_ROOT / "scratch"


def expand_config_path(value: str) -> str:
    """Expand `${VAR}` in a config string; `${PROJECT_SCRATCH}` always resolves."""

    environment = dict(os.environ)
    environment.setdefault("PROJECT_SCRATCH", scratch_root().as_posix())
    expanded = os.path.expandvars(_substitute(value, environment))
    if "${" in expanded:
        raise ValueError(f"config references an unset environment variable: {value}")
    return expanded


def _substitute(value: str, environment: dict[str, str]) -> str:
    return value.replace("${PROJECT_SCRATCH}", environment["PROJECT_SCRATCH"])


def default_clip_weights_root() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home) / "clip"
    return scratch_root() / "data" / "clip-weights"


def auto_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"
