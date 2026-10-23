#!/usr/bin/env python
"""Working demo: BioCLIP zero-shot species classification against the 400 Rare Species labels.

Downloads the pinned BioCLIP checkpoint (~600 MB, cached) and the Rare Species label
metadata (~3 MB). Images come either from your own files or, with ``--sample N``,
streamed from the Rare Species dataset on the Hugging Face Hub (reads ~250 MB).

    python scripts/demo.py --sample 6                 # 6 dataset images, one every 200 rows, with ground truth
    python scripts/demo.py --image photo.jpg [...]    # your own photos
    python scripts/demo.py --sample 6 --prompt-style common --prefix "a photo of "

The label set is the Rare Species species list; a photo of a species outside that
list will still be ranked against those 400 names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from biodiv.clip_eval import classify_images
from biodiv.encoders import KNOWN_CHECKPOINTS, OpenClipEncoder
from biodiv.paths import get_paths
from biodiv.prompts import PROMPT_STYLES
from biodiv.species_data import rare_species_stream, rare_species_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", type=Path, default=[], help="image file to classify (repeatable)")
    ap.add_argument("--sample", type=int, default=0, help="stream N Rare Species images from the Hub instead")
    ap.add_argument("--stride", type=int, default=200, help="rows between streamed samples (30 rows per species)")
    ap.add_argument("--checkpoint", default="imageomics/bioclip", help="Hub repo id, optionally org/repo@revision")
    ap.add_argument("--prompt-style", default="taxonomic_common", choices=PROMPT_STYLES)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None, help="overrides BIODIV_CACHE_DIR")
    args = ap.parse_args()
    if not args.image and not args.sample:
        ap.error("give --image FILE (repeatable) and/or --sample N")

    cache = str(get_paths(cache_dir=args.cache_dir).cache_dir)
    repo_id, _, revision = args.checkpoint.partition("@")
    encoder = OpenClipEncoder(repo_id, revision or KNOWN_CHECKPOINTS.get(repo_id), cache, args.device)
    table = rare_species_table(cache)
    print(f"checkpoint {repo_id}@{(revision or KNOWN_CHECKPOINTS.get(repo_id) or 'latest')[:12]} on {encoder.device}; "
          f"{len(table)} species labels; prompt style {args.prompt_style!r}, prefix {args.prefix!r}")

    items = [(f"image {p}", Image.open(p), None) for p in args.image]
    if args.sample:
        items += [(f"dataset row {i * args.stride}", im, sci) for i, (im, sci) in enumerate(rare_species_stream(args.sample, args.stride, cache))]

    rankings = classify_images(encoder, [im for _, im, _ in items], table, args.prompt_style, args.prefix, args.top_k)
    correct = 0
    for (name, _, truth), ranked in zip(items, rankings):
        print(f"\n{name}" + (f"   true: {truth} ({_common(table, truth)})" if truth else ""))
        for rank, (species, score) in enumerate(ranked, 1):
            mark = "  <- correct" if truth and species.scientific == truth else ""
            print(f"  {rank}. {species.scientific:28s} {species.common or '':30s} cosine {score:.3f}{mark}")
        correct += bool(truth) and ranked[0][0].scientific == truth
    if args.sample:
        print(f"\ntop-1 correct on these {args.sample} streamed samples: {correct}/{args.sample} (a demo, not an evaluation)")


def _common(table, scientific):
    return next((s.common for s in table if s.scientific == scientific), "")


if __name__ == "__main__":
    main()
