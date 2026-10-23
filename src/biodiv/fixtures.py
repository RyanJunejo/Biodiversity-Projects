"""Tiny synthetic fixtures for tests and smoke runs: no real images, no
downloads, no model weights. They live in the package (not under ``tests/``)
so that ``scripts/eval_bioclip.py --dataset fixture`` and the notebooks' smoke
mode work after a plain ``pip install``.

``make_inat_fixture`` writes an iNaturalist-style split (images + JSON) with
deliberately shuffled, non-contiguous category ids.

``load_fixture_rare_species`` returns in-memory records and a species table
with four species, two of which share a common name, mirroring the real
Rare Species collisions. ``fixture_encoder`` returns a ``LookupEncoder`` whose
image embeddings are one-hot species vectors and whose text embeddings map
prompts back to those vectors, so predictions are fully predictable.

``python -m biodiv.fixtures [DIR]`` materialises the iNat fixture (default
``tests/fixtures/data``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

# (category_id, name, colour); ids are shuffled and non-contiguous on purpose.
INAT_CATEGORIES = [
    (42, "Amanita muscaria", (200, 30, 30)),
    (7, "Cicindela campestris", (30, 200, 30)),
    (3, "Lumbricus terrestris", (30, 30, 200)),
]


def make_image(colour, size=32, marker=True) -> Image.Image:
    img = Image.new("RGB", (size, size), colour)
    if marker:  # asymmetric marker so a horizontal flip changes the pixels
        ImageDraw.Draw(img).rectangle([0, 0, size // 8, size // 8], fill=(255, 255, 255))
    return img


def make_inat_fixture(root: str | Path, n_train_per_class: int = 2, n_val_per_class: int = 1, image_size: int = 32, seed: int = 0) -> dict:
    root = Path(root)
    rng = random.Random(seed)
    categories = [{"id": cid, "name": name, "kingdom": "Fixture", "common_name": name.split()[0]} for cid, name, _ in INAT_CATEGORIES]
    rng.shuffle(categories)  # JSON order != id order
    paths = {}
    for split, per_class in (("train", n_train_per_class), ("val", n_val_per_class)):
        images, annotations, next_id = [], [], 1000 if split == "train" else 5000
        for cid, name, colour in INAT_CATEGORIES:
            for k in range(per_class):
                file_name = f"images/{split}/{cid}/{k}.png"
                (root / file_name).parent.mkdir(parents=True, exist_ok=True)
                make_image(colour, image_size).save(root / file_name)
                images.append({"id": next_id, "file_name": file_name, "width": image_size, "height": image_size})
                annotations.append({"id": next_id, "image_id": next_id, "category_id": cid})
                next_id += 1
        rng.shuffle(images)  # image order != annotation order
        data = {"info": {"fixture": True}, "images": images, "annotations": annotations, "categories": categories, "licenses": []}
        (root / f"{split}.json").write_text(json.dumps(data, indent=1))
        paths[split] = root / f"{split}.json"
    (root / "categories.json").write_text(json.dumps(categories))
    paths["categories"] = root / "categories.json"
    paths["image_root"] = root
    return paths


# Rare-species-like fixture: 4 species, 3 unique common names.
def _row(sci, common, phylum, cls, order, family, colour):
    genus, species = sci.split()
    return {"sciName": sci, "common": common, "kingdom": "Animalia", "phylum": phylum, "class": cls, "order": order,
            "family": family, "genus": genus, "species": species, "colour": colour}


RARE_SPECIES_ROWS = [
    _row("Acropora acuminata", "staghorn coral", "Cnidaria", "Anthozoa", "Scleractinia", "Acroporidae", (255, 0, 0)),
    _row("Acropora millepora", "staghorn coral", "Cnidaria", "Anthozoa", "Scleractinia", "Acroporidae", (0, 255, 0)),
    _row("Dolomedes plantarius", "Fen raft spider", "Arthropoda", "Arachnida", "Araneae", "Pisauridae", (0, 0, 255)),
    _row("Tylototriton shanjing", "Yunnan Newt", "Chordata", "Amphibia", "Caudata", "Salamandridae", (255, 255, 0)),
]


def load_fixture_rare_species(images_per_species: int = 3):
    from .species_data import rare_species_from_rows

    rows = []
    for r in RARE_SPECIES_ROWS:
        for _ in range(images_per_species):
            rows.append({**{k: v for k, v in r.items() if k != "colour"}, "image": make_image(r["colour"], 16, marker=False)})
    return rare_species_from_rows(rows, image_key="image")


def fixture_encoder(dim: int = 8):
    """LookupEncoder with predictable geometry.

    Each fixture species gets a one-hot vector. Image embeddings map an image's
    mean colour to the nearest species' vector; text embeddings average the
    vectors of every species named in the prompt, so a prompt shared by two
    species sits exactly between them and produces an exact score tie.
    """
    from .encoders import LookupEncoder

    colours = torch.tensor([r["colour"] for r in RARE_SPECIES_ROWS], dtype=torch.float32) / 255.0
    one_hot = torch.eye(dim)[: len(RARE_SPECIES_ROWS)]

    def image_fn(images: torch.Tensor) -> torch.Tensor:
        mean_rgb = images.mean(dim=(2, 3))  # [n, 3] in 0..1
        nearest = torch.cdist(mean_rgb, colours).argmin(dim=1)
        return one_hot[nearest]

    def text_vec(prompt: str) -> torch.Tensor:
        by_sci = [i for i, r in enumerate(RARE_SPECIES_ROWS) if r["sciName"] in prompt]
        matches = by_sci or [i for i, r in enumerate(RARE_SPECIES_ROWS) if r["common"] in prompt]
        if not matches:
            raise KeyError(f"fixture encoder cannot place prompt {prompt!r}")
        return one_hot[matches].mean(dim=0)

    enc = LookupEncoder(dim=dim, image_fn=image_fn)
    enc.encode_text = lambda texts: torch.stack([text_vec(t) for t in texts])  # type: ignore[method-assign]
    return enc


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/data")
    paths = make_inat_fixture(out)
    print({k: str(v) for k, v in paths.items()})
