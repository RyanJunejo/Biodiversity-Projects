"""End-to-end smoke test of the training loop on the synthetic fixture (CPU).

This checks plumbing only: the model is randomly initialised and the data is
synthetic, so nothing here reproduces a 2024 experiment.
"""

import json

import pytest
import torch

hiera = pytest.importorskip("hiera")

from biodiv.paths import get_paths  # noqa: E402
from biodiv.train import TrainConfig, run_training  # noqa: E402


def test_train_then_resume(inat_fixture, tmp_path):
    cfg = TrainConfig.from_yaml("tests/fixtures/hiera_smoke.yaml")
    cfg.output_dir = str(tmp_path / "run")
    paths = get_paths(data_root=inat_fixture["image_root"], output_dir=tmp_path)
    logs = []
    first = run_training(cfg, paths, log=logs.append)
    assert first["start_epoch"] == 0 and first["epochs_run"] == 2
    assert (tmp_path / "run" / "checkpoints" / "epoch_0001.pt").exists()
    assert all(0.0 <= h["val_top1"] <= 1.0 for h in first["history"])

    before = {k: v.clone() for k, v in torch.load(tmp_path / "run" / "final_model.pt").items()}
    cfg.epochs = 3  # extend the finished run; auto-resume should pick up after epoch 1
    second = run_training(cfg, paths, log=logs.append)
    assert second["start_epoch"] == 2 and second["epochs_run"] == 1
    assert any("resumed from" in line for line in logs)
    # the learning rate must come from the *new* 3-epoch schedule, not the old schedule's terminal 0.0,
    # and the resumed epoch must actually update the weights
    assert second["history"][0]["lr_head"] > 0
    after = torch.load(tmp_path / "run" / "final_model.pt")
    assert any(not torch.equal(before[k], after[k]) for k in before)
    history = [json.loads(l) for l in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in history] == [0, 1, 2]
