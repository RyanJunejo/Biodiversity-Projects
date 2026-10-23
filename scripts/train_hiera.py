#!/usr/bin/env python
"""Fine-tune Hiera on an iNaturalist competition split.

Examples
--------
    # full run (needs the dataset under $BIODIV_DATA_ROOT and a GPU)
    python scripts/train_hiera.py --config configs/hiera_inat2019.yaml

    # resume automatically from outputs/<run_name>/checkpoints/last.pt
    python scripts/train_hiera.py --config configs/hiera_inat2019.yaml --resume auto

    # smoke test on the synthetic fixture (CPU, seconds)
    python -m biodiv.fixtures tests/fixtures/data
    python scripts/train_hiera.py --config tests/fixtures/hiera_smoke.yaml --data-root tests/fixtures/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biodiv.paths import get_paths
from biodiv.train import TrainConfig, run_training


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-root", default=None, help="overrides BIODIV_DATA_ROOT")
    ap.add_argument("--output-dir", default=None, help="overrides BIODIV_OUTPUT_DIR")
    ap.add_argument("--resume", default=None, help="'auto', a checkpoint path, or 'none'")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.resume is not None:
        cfg.resume = None if args.resume == "none" else args.resume
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.max_batches is not None:
        cfg.max_batches_per_epoch = args.max_batches
    if args.wandb:
        cfg.wandb["enabled"] = True
    summary = run_training(cfg, get_paths(data_root=args.data_root, output_dir=args.output_dir))
    print(f"done: {summary['epochs_run']} epoch(s) written to {summary['output_dir']}")


if __name__ == "__main__":
    main()
