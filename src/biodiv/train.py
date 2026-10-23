"""Config-driven Hiera fine-tuning on iNaturalist competition data.

One training loop serves both the 2019 and the 2021-mini experiments; the
dataset-specific parts live in ``configs/*.yaml``. See ``docs/changes.md``
for how this differs from the 2024 scripts.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import checkpoint as ckpt
from .hiera_model import build_hiera
from .inat import INatDataset, build_records, load_split
from .optim import build_scheduler, layerwise_lr_param_groups
from .paths import Paths, get_paths, resolve
from .transforms import build_eval_transform, build_train_transform


@dataclass
class TrainConfig:
    # dataset (paths relative to BIODIV_DATA_ROOT unless absolute)
    dataset_name: str = "inat"
    train_json: str = ""
    val_json: str = ""
    image_root: str = ""
    categories_json: str | None = None
    image_size: int = 224
    batch_size: int = 512
    eval_batch_size: int = 512
    num_workers: int = 8
    # model
    model_name: str = "hiera_tiny_224"
    model_checkpoint: str = "mae_in1k"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    # optimisation
    epochs: int = 300
    warmup_epochs: int = 5
    base_lr: float = 2e-3
    layer_decay: float = 0.65
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    amp: bool = True
    # augmentation
    horizontal_flip: bool = True
    randaugment_num_ops: int = 9
    randaugment_magnitude: int = 1
    # run
    run_name: str = "hiera_run"
    output_dir: str | None = None  # defaults to BIODIV_OUTPUT_DIR / run_name
    checkpoint_every: int = 10
    resume: str | None = "auto"  # "auto", a path, or None
    seed: int = 0
    max_batches_per_epoch: int | None = None  # smoke tests only
    wandb: dict = field(default_factory=lambda: {"enabled": False, "project": None, "entity": None})

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.betas = tuple(cfg.betas)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg: TrainConfig, paths: Paths):
    train_data, train_map = load_split(resolve(cfg.train_json, paths.data_root), cfg.categories_json and resolve(cfg.categories_json, paths.data_root))
    val_data, val_map = load_split(resolve(cfg.val_json, paths.data_root), cfg.categories_json and resolve(cfg.categories_json, paths.data_root))
    train_map.assert_same(val_map)
    root = resolve(cfg.image_root, paths.data_root)
    train_tf = build_train_transform(cfg.image_size, cfg.horizontal_flip, cfg.randaugment_num_ops, cfg.randaugment_magnitude)
    train_ds = INatDataset(build_records(train_data, train_map), root, train_tf)
    val_ds = INatDataset(build_records(val_data, val_map), root, build_eval_transform(cfg.image_size))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, train_map


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict:
    """Validation loss on logits plus top-1 / top-5 accuracy."""
    model.eval()
    total_loss, correct1, correct5, n = 0.0, 0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, labels).item() * inputs.size(0)
        top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        n += inputs.size(0)
    return {"val_loss": total_loss / max(n, 1), "val_top1": correct1 / max(n, 1), "val_top5": correct5 / max(n, 1), "val_images": n}


def train_one_epoch(model, loader, optimizer, scaler, criterion, mix, device, amp: bool, max_batches: int | None = None) -> dict:
    model.train()
    running, n = 0.0, 0
    for step, (inputs, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = mix(inputs, labels)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return {"train_loss": running / max(n, 1), "train_images": n}


def run_training(cfg: TrainConfig, paths: Paths | None = None, log: Callable[[str], None] = print) -> dict:
    paths = paths or get_paths()
    set_seed(cfg.seed)
    device = pick_device()
    out_dir = Path(cfg.output_dir) if cfg.output_dir else paths.output_dir / cfg.run_name
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    train_loader, val_loader, cat_map = build_dataloaders(cfg, paths)
    num_classes = cat_map.num_classes
    log(f"device={device} classes={num_classes} train_images={len(train_loader.dataset)} val_images={len(val_loader.dataset)}")

    model = build_hiera(num_classes, cfg.model_name, cfg.model_checkpoint, cfg.drop_path_rate, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(layerwise_lr_param_groups(model, cfg.base_lr, cfg.layer_decay, cfg.weight_decay), betas=cfg.betas)
    scheduler = build_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    mix = v2.RandomChoice([v2.CutMix(alpha=cfg.cutmix_alpha, num_classes=num_classes), v2.MixUp(alpha=cfg.mixup_alpha, num_classes=num_classes)])

    start_epoch = 0
    resume_path = ckpt.find_latest(ckpt_dir) if cfg.resume == "auto" else (Path(cfg.resume) if cfg.resume else None)
    if resume_path is not None and Path(resume_path).exists():
        meta = ckpt.load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=str(device))
        start_epoch = meta["epoch"] + 1
        log(f"resumed from {resume_path} (completed epoch {meta['epoch']}), starting at epoch {start_epoch}")

    wandb_run = None
    if cfg.wandb.get("enabled"):
        import wandb

        wandb_run = wandb.init(project=cfg.wandb.get("project"), entity=cfg.wandb.get("entity"), name=cfg.run_name, config=asdict(cfg), resume="allow")

    history = []
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        stats = train_one_epoch(model, train_loader, optimizer, scaler, criterion, mix, device, cfg.amp, cfg.max_batches_per_epoch)
        stats.update(evaluate(model, val_loader, device, criterion))
        stats.update({"epoch": epoch, "lr_head": optimizer.param_groups[-1]["lr"], "epoch_seconds": time.time() - t0})
        scheduler.step()
        history.append(stats)
        log(json.dumps(stats))
        if wandb_run is not None:
            wandb_run.log(stats, step=epoch)
        if (epoch + 1) % cfg.checkpoint_every == 0 or epoch + 1 == cfg.epochs:
            ckpt.save_checkpoint(ckpt_dir, epoch, model, optimizer, scheduler, scaler, extra={"config": asdict(cfg)})
        with (out_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(stats) + "\n")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    if wandb_run is not None:
        wandb_run.finish()
    return {"output_dir": str(out_dir), "start_epoch": start_epoch, "epochs_run": len(history), "history": history}
