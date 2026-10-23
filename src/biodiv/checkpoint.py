"""Checkpoint save / resume that does not depend on SLURM environment variables.

The 2024 scripts built the checkpoint directory from ``$USER`` and
``$SLURM_JOB_ID`` (``os.path.join`` raises when either is unset), used
different file names in the two scripts, and had the resume branch commented
out. Here the directory is explicit, files are named ``epoch_{n:04d}.pt``
with a ``last.pt`` copy, and ``find_latest`` locates the newest one.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch

_EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")


def save_checkpoint(
    checkpoint_dir: str | Path,
    epoch: int,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    extra: dict | None = None,
) -> Path:
    """Save state after finishing ``epoch`` (0-based); returns the file path."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save(state, path)
    shutil.copyfile(path, checkpoint_dir / "last.pt")
    return path


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    """Restore state in place and return the checkpoint metadata.

    Returns a dict with ``epoch`` (the last completed epoch, 0-based), so
    training should resume from ``epoch + 1``.
    """
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return {"epoch": int(state["epoch"]), "extra": state.get("extra", {})}


def find_latest(checkpoint_dir: str | Path) -> Path | None:
    """Newest ``epoch_*.pt`` in ``checkpoint_dir`` by epoch number, if any."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    best, best_epoch = None, -1
    for p in checkpoint_dir.glob("epoch_*.pt"):
        m = _EPOCH_RE.search(p.name)
        if m and int(m.group(1)) > best_epoch:
            best, best_epoch = p, int(m.group(1))
    return best
