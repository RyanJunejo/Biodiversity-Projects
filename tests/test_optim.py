"""Layer-wise learning-rate groups and the warmup + cosine schedule."""

import math

import torch
import torch.nn as nn

from biodiv.optim import build_scheduler, layer_id_for, layerwise_lr_param_groups, warmup_cosine_factor


class FakeHiera(nn.Module):
    def __init__(self, num_blocks=3):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.pos_embed = nn.Parameter(torch.zeros(2))
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(2)
        self.head = nn.Linear(2, 2)


def test_layer_ids_and_decayed_lrs():
    m = FakeHiera(3)
    groups = layerwise_lr_param_groups(m, base_lr=1e-3, layer_decay=0.5, weight_decay=0.05)
    lrs = {g["layer_id"]: g["lr"] for g in groups}
    assert lrs[4] == 1e-3  # head + norm
    assert math.isclose(lrs[3], 1e-3 * 0.5)  # last block
    assert math.isclose(lrs[1], 1e-3 * 0.5 ** 3)  # first block
    assert math.isclose(lrs[0], 1e-3 * 0.5 ** 4)  # embeddings
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))
    assert layer_id_for("blocks.11.attn.qkv.weight", 12) == 12 and layer_id_for("pos_embed", 12) == 0


def test_warmup_then_cosine_to_zero():
    f = [warmup_cosine_factor(e, warmup_epochs=5, total_epochs=20) for e in range(20)]
    assert f[:5] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert f[5] == 1.0 and f[6] < f[5]
    assert math.isclose(f[-1], 0.5 * (1 + math.cos(math.pi * 14 / 15)))
    assert all(a >= b for a, b in zip(f[5:], f[6:]))


def test_scheduler_drives_group_lrs():
    m = FakeHiera(2)
    opt = torch.optim.AdamW(layerwise_lr_param_groups(m, 1e-2, 0.5, 0.0))
    sched = build_scheduler(opt, warmup_epochs=2, total_epochs=4)
    seen = []
    for _ in range(4):
        seen.append(opt.param_groups[-1]["lr"])
        opt.step()
        sched.step()
    assert math.isclose(seen[0], 5e-3) and math.isclose(seen[1], 1e-2) and math.isclose(seen[2], 1e-2) and math.isclose(seen[3], 5e-3)


def test_scheduler_resumes_at_a_given_epoch_from_initial_lr():
    m = FakeHiera(2)
    opt = torch.optim.AdamW(layerwise_lr_param_groups(m, 1e-2, 0.5, 0.0))
    for g in opt.param_groups:
        g["lr"] = 0.0  # what a checkpoint of a finished run would restore
        g["initial_lr"] = g["lr"] if False else None
    base = [1e-2 * 0.5 ** (3 - g["layer_id"]) for g in opt.param_groups]
    for g, b in zip(opt.param_groups, base):
        g["initial_lr"] = b
    build_scheduler(opt, warmup_epochs=2, total_epochs=6, last_epoch=2)  # about to run epoch 3
    expected = warmup_cosine_factor(3, 2, 6)
    assert all(math.isclose(g["lr"], b * expected) for g, b in zip(opt.param_groups, base))
    assert expected > 0
