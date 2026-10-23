#!/usr/bin/env python
"""Zero-shot evaluation of BioCLIP-style checkpoints.

Examples
--------
    # BioCLIP on the iNat21 validation split with the 2024 prompt format
    python scripts/eval_bioclip.py --dataset inat21-val --checkpoint bioclip \
        --prompt-style taxonomic_common --prefix "" --output results/runs/bioclip_inat21_val.json

    # Rare Species, all prompt styles, both tie policies are always reported
    python scripts/eval_bioclip.py --dataset rare-species --checkpoint bioclip \
        --prompt-style common --prefix "a photo of "

    # any Hub checkpoint, optionally pinned
    python scripts/eval_bioclip.py --dataset rare-species --checkpoint imageomics/bioclip@<commit>

    # smoke test on the synthetic fixture with a weight-free encoder
    python scripts/eval_bioclip.py --dataset fixture --checkpoint lookup --prompt-style common
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from biodiv.clip_eval import evaluate_zero_shot
from biodiv.encoders import KNOWN_CHECKPOINTS, OpenClipEncoder
from biodiv.paths import get_paths, resolve
from biodiv.prompts import PROMPT_STYLES
from biodiv.species_data import inat21_val, rare_species_from_hub, rare_species_from_jsonl


def load_dataset(name: str, cfg: dict, paths):
    if name == "fixture":
        from tests.fixtures.make_fixture import load_fixture_rare_species

        return load_fixture_rare_species()
    if name in ("inat21-val", "inat2019-val"):
        d = cfg["datasets"][name]
        from biodiv.inat import load_json

        val_json = resolve(d["val_json"], paths.data_root)
        if d.get("categories_json"):  # iNat2019: real names live in a separate file
            data = load_json(val_json)
            data["categories"] = load_json(resolve(d["categories_json"], paths.data_root))
            tmp = val_json.parent / (val_json.stem + ".with_names.json")
            tmp.write_text(json.dumps(data))
            val_json = tmp
        return inat21_val(val_json, resolve(d["image_root"], paths.data_root))
    if name == "rare-species":
        d = cfg["datasets"][name]
        if d.get("source", "hub") == "hub":
            return rare_species_from_hub(cache_dir=str(paths.cache_dir), revision=d.get("revision"))
        return rare_species_from_jsonl(resolve(d["source"], paths.data_root))
    raise SystemExit(f"unknown dataset {name!r}")


def load_encoder(spec: str, cfg: dict, paths, device: str | None):
    if spec == "lookup":  # weight-free stand-in with predictable geometry for the fixture dataset
        from tests.fixtures.make_fixture import fixture_encoder

        return fixture_encoder()
    if spec in cfg.get("checkpoints", {}):
        c = cfg["checkpoints"][spec]
        return OpenClipEncoder(c["repo_id"], c.get("revision"), str(paths.cache_dir), device)
    repo_id, _, revision = spec.partition("@")
    return OpenClipEncoder(repo_id, revision or KNOWN_CHECKPOINTS.get(repo_id), str(paths.cache_dir), device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/bioclip_eval.yaml"))
    ap.add_argument("--dataset", required=True, help="inat21-val | inat2019-val | rare-species | fixture")
    ap.add_argument("--checkpoint", required=True, help="name from config, org/repo[@revision], or 'lookup'")
    ap.add_argument("--prompt-style", default="taxonomic_common", choices=PROMPT_STYLES)
    ap.add_argument("--prefix", default="", help='e.g. "a photo of "')
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", type=Path, default=None, help="write the result JSON here")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    paths = get_paths(data_root=args.data_root, cache_dir=args.cache_dir)
    records, table = load_dataset(args.dataset, cfg, paths)
    if args.limit:
        records = records[: args.limit]
    encoder = load_encoder(args.checkpoint, cfg, paths, args.device)
    result = evaluate_zero_shot(
        encoder, records, table, args.prompt_style, args.prefix,
        batch_size=args.batch_size or cfg.get("batch_size", 32),
        num_workers=args.num_workers if args.num_workers is not None else cfg.get("num_workers", 0),
    )
    result.update({"dataset": args.dataset, "checkpoint": args.checkpoint, "limit": args.limit})
    print(json.dumps({k: v for k, v in result.items() if k != "skipped"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
