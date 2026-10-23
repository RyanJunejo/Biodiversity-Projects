"""Checkpoint resumption restores model, optimiser, scheduler and epoch."""

import torch
import torch.nn as nn

from biodiv.checkpoint import find_latest, load_checkpoint, save_checkpoint
from biodiv.optim import build_scheduler


def _setup(seed):
    torch.manual_seed(seed)
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = build_scheduler(opt, warmup_epochs=2, total_epochs=10)
    scaler = torch.amp.GradScaler(enabled=False)
    return model, opt, sched, scaler


def test_resume_restores_training_state(tmp_path):
    model, opt, sched, scaler = _setup(0)
    for epoch in range(3):
        loss = model(torch.randn(8, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        save_checkpoint(tmp_path, epoch, model, opt, sched, scaler, extra={"note": "t"})
    assert find_latest(tmp_path).name == "epoch_0002.pt" and (tmp_path / "last.pt").exists()

    fresh_model, fresh_opt, fresh_sched, fresh_scaler = _setup(123)
    assert not torch.equal(fresh_model.weight, model.weight)
    meta = load_checkpoint(find_latest(tmp_path), fresh_model, fresh_opt, fresh_sched, fresh_scaler)
    assert meta["epoch"] == 2 and meta["extra"] == {"note": "t"}
    assert torch.equal(fresh_model.weight, model.weight)
    assert fresh_sched.last_epoch == sched.last_epoch == 3
    assert fresh_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    s_new, s_old = fresh_opt.state_dict()["state"], opt.state_dict()["state"]
    assert all(torch.equal(s_new[k]["exp_avg"], s_old[k]["exp_avg"]) for k in s_old)


def test_find_latest_without_directory(tmp_path):
    assert find_latest(tmp_path / "missing") is None
