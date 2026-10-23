"""Adapters from dataset metadata to ``(records, SpeciesTable)`` pairs.

* iNat21 validation split: ``val.json`` categories -> species table; each
  annotation's ``category_id`` is mapped to a ``species_id`` explicitly.
* Rare Species (imageomics/rare-species on the Hugging Face Hub): rows carry
  ``sciName``, ``common`` and the seven taxonomic ranks. 400 species, 11,983
  images (30 per species nominally; the dataset card notes 5 species have
  fewer than 30 unique images). Only 398 common names are unique: Acropora
  acuminata / A. millepora share "staghorn coral" and Tylototriton shanjing /
  T. verrucosus share "Yunnan Newt".
"""

from __future__ import annotations

import json
from pathlib import Path

from .clip_eval import SpeciesTable
from .inat import load_json

RARE_SPECIES_REPO = "imageomics/rare-species"
RARE_SPECIES_REVISION = "7e984e1f68ab70170e7913286ad1ff1a0e5ac370"  # Hub commit observed 2026-10-23


def inat21_val(val_json: str | Path, image_root: str | Path | None = None):
    """``(records, table)`` for an iNat21 split file; records are ``(file_name, species_id)``."""
    data = load_json(val_json)
    table = SpeciesTable.from_inat_categories(data["categories"])
    images = {img["id"]: img["file_name"] for img in data["images"]}
    records = []
    for ann in data["annotations"]:
        file_name = images[ann["image_id"]]
        if image_root is not None:
            file_name = str(Path(image_root) / file_name)
        records.append((file_name, table.species_id_for(int(ann["category_id"]))))
    return records, table


def rare_species_from_rows(rows: list[dict], image_key: str = "image"):
    """``rows`` are dicts with ``sciName``, ``common``, the 7 ranks and an image (path or PIL)."""
    table = SpeciesTable.from_rows(rows, key="sciName", common_key="common")
    records = [(r[image_key], table.species_id_for(str(r["sciName"]))) for r in rows]
    return records, table


def rare_species_from_jsonl(path: str | Path):
    """Rows exported with ``datasets.Dataset.to_json`` (image column holds ``{"path": ...}``)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                img = r["image"]
                r["image"] = img["path"] if isinstance(img, dict) else img
                rows.append(r)
    return rare_species_from_rows(rows)


def rare_species_from_hub(cache_dir: str | None = None, revision: str | None = RARE_SPECIES_REVISION, split: str = "train"):
    """Load from the Hub with ``datasets`` (optional dependency); images are PIL objects."""
    from datasets import load_dataset

    ds = load_dataset(RARE_SPECIES_REPO, split=split, cache_dir=cache_dir, revision=revision)
    rows = [dict(r) for r in ds]
    return rare_species_from_rows(rows, image_key="image")
