import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def configure_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    If ``deterministic`` is True, enable deterministic cuBLAS operations which
    may incur a performance penalty but improves reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.use_deterministic_algorithms(True)


def get_commit_hash() -> Optional[str]:
    """Return the current Git commit hash if available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_config(
    path: Path, args: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> None:
    """Save run configuration to ``path`` in YAML format."""

    if isinstance(args, argparse.Namespace):
        cfg: Dict[str, Any] = vars(args).copy()
    else:
        cfg = dict(args)

    if extra:
        cfg.update({k: v for k, v in extra.items() if v is not None})

    commit = get_commit_hash()
    if commit:
        cfg["commit"] = commit

    def _convert(v: Any) -> Any:
        if isinstance(v, Path):
            return v.as_posix()
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(_convert(val) for val in v)
        return v

    cfg = _convert(cfg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
