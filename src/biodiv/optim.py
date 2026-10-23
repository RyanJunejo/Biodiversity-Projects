"""Optimiser parameter groups and learning-rate schedule for Hiera fine-tuning.

Layer-wise learning-rate decay follows the usual MAE / BEiT fine-tuning
convention: the head and final norm get ``base_lr``; block ``i`` of ``L``
blocks gets ``base_lr * decay ** (L - i)``; the patch embedding and position
embedding get ``base_lr * decay ** (L + 1)``.

The 2024 scripts implemented a different, name-parsing version of this idea.
Replaying that loop on hiera_tiny_224 (12 blocks, decay 0.65, start 3e-3)
gives: head 1.95e-3, final norm 1.27e-3, block 11 8.2e-4 ... block 0 7.2e-6,
patch/pos embed 4.7e-6. This module, with ``base_lr=2e-3``, gives:
head and norm 2e-3, block 11 1.3e-3 ... block 0 1.1e-5, patch/pos embed
7.2e-6. The AdamW ``lr`` argument of the 2024 scripts (2e-3) was never used
because every parameter sat in its own group with an explicit ``lr``.

The schedule is linear warmup followed by cosine decay to zero over the
remaining epochs, stepped once per epoch. The 2024 scripts used a constant
0.1x factor for the first 5 epochs (not a ramp) and a cosine with
``T_max=300`` that was only stepped 295 times.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def layer_id_for(name: str, num_blocks: int) -> int:
    """0 for embeddings, 1..L for blocks, L+1 for the final norm and head."""
    if name.startswith(("patch_embed", "pos_embed")):
        return 0
    m = _BLOCK_RE.match(name)
    if m:
        return int(m.group(1)) + 1
    return num_blocks + 1


def layerwise_lr_param_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> list[dict]:
    """One parameter group per layer id, with geometrically decayed ``lr``."""
    num_blocks = len(model.blocks)
    groups: dict[int, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = layer_id_for(name, num_blocks)
        if lid not in groups:
            scale = layer_decay ** (num_blocks + 1 - lid)
            groups[lid] = {"params": [], "lr": base_lr * scale, "weight_decay": weight_decay, "layer_id": lid}
        groups[lid]["params"].append(param)
    return [groups[k] for k in sorted(groups)]


def warmup_cosine_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Multiplier applied to each group's base lr at the start of ``epoch``."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max(epoch - warmup_epochs, 0) / decay_epochs, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, last_epoch: int = -1) -> LambdaLR:
    """Per-epoch LambdaLR. Pass ``last_epoch=start_epoch - 1`` when resuming so
    the groups' ``lr`` is recomputed for ``start_epoch`` from the *current*
    config (every group must carry ``initial_lr``)."""
    return LambdaLR(optimizer, lambda e: warmup_cosine_factor(e, warmup_epochs, total_epochs), last_epoch=last_epoch)
