"""Zero-shot species classification with CLIP-style encoders.

Design points, and how they differ from the 2024 notebooks:

* Species identity (``species_id``) is separate from prompt text. Two species
  with the same common name keep separate ids; their identical prompts produce
  identical text embeddings and therefore exact score ties, which are handled
  by an explicit ``tie_policy`` instead of silently merging the species.
* Labels are mapped through an explicit table (``SpeciesTable``) rather than
  assuming prediction index == dataset category id.
* Image and text embeddings are both L2-normalised, so scores are cosine
  similarities (the 2024 iNat21 notebook left the text side unnormalised).
* Each image travels with its label and record index through the data loader,
  and unreadable images are skipped and reported, so predictions and labels
  cannot drift out of alignment.
* No unconditional CUDA calls; CPU execution works.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .metrics import accuracy, topk_accuracy
from .prompts import RANKS, make_prompts


@dataclass(frozen=True)
class Species:
    species_id: int
    scientific: str
    common: str | None
    taxonomy: dict = field(default_factory=dict)  # kingdom .. genus, species (epithet)
    source_id: int | str | None = None  # e.g. iNat category id


class SpeciesTable:
    """Ordered list of species; position == species_id."""

    def __init__(self, species: Sequence[Species]):
        for i, s in enumerate(species):
            if s.species_id != i:
                raise ValueError(f"species_id {s.species_id} at position {i}")
        self.species = list(species)
        self._by_source = {s.source_id: s.species_id for s in self.species if s.source_id is not None}
        if len(self._by_source) != sum(s.source_id is not None for s in self.species):
            raise ValueError("duplicate source ids in species table")

    @classmethod
    def from_inat_categories(cls, categories: Sequence[dict]) -> "SpeciesTable":
        """iNaturalist ``categories`` entries -> table ordered by category id.

        Works with both competition formats. iNat 2021 entries carry
        ``specific_epithet`` and ``common_name``; iNat 2019 entries (from the
        separate ``categories.json``) have only the six ranks and ``name``
        (``"Genus epithet"``), so the epithet is taken from ``name`` and the
        common name is ``None`` (common-name prompts then raise).
        """
        rows = []
        for i, c in enumerate(sorted(categories, key=lambda c: int(c["id"]))):
            name = str(c["name"])
            taxonomy = {r: c.get(r) for r in RANKS[:-1]}
            epithet = c.get("specific_epithet")
            if epithet is None:
                parts = name.split()
                epithet = " ".join(parts[1:]) if len(parts) > 1 else ""
            taxonomy["species"] = epithet
            rows.append(Species(i, name, c.get("common_name"), taxonomy, int(c["id"])))
        return cls(rows)

    @classmethod
    def from_rows(cls, rows: Iterable[dict], key: str = "sciName", common_key: str = "common") -> "SpeciesTable":
        """Unique species from per-image metadata rows (e.g. Rare Species).

        Species are identified by ``key`` (scientific name) and ordered
        alphabetically; the common name and taxonomy are taken from the first
        row seen for each species.
        """
        seen: dict[str, dict] = {}
        for r in rows:
            seen.setdefault(str(r[key]), r)
        species = []
        for i, name in enumerate(sorted(seen)):
            r = seen[name]
            species.append(Species(i, name, r.get(common_key), {rk: r[rk] for rk in RANKS}, name))
        return cls(species)

    def __len__(self) -> int:
        return len(self.species)

    def __iter__(self):
        return iter(self.species)

    def __getitem__(self, i: int) -> Species:
        return self.species[i]

    def species_id_for(self, source_id) -> int:
        return self._by_source[source_id]

    def prompts(self, style: str, prefix: str = "") -> list[str]:
        return make_prompts(self.species, style, prefix)


def duplicate_prompt_groups(prompts: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        groups.setdefault(p, []).append(i)
    return {p: ids for p, ids in groups.items() if len(ids) > 1}


class ImageRecords(Dataset):
    """``records`` are ``(source, species_id)``.

    ``source`` is a path, a PIL image, or a zero-argument callable returning a
    PIL image (used for lazily decoded Hub datasets).
    """

    def __init__(self, records: Sequence[tuple], preprocess: Callable, root: str | Path | None = None):
        self.records = list(records)
        self.preprocess = preprocess
        self.root = Path(root) if root else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        source, label = self.records[idx]
        try:
            if callable(source):
                source = source()
            if isinstance(source, Image.Image):
                image = source
            else:
                path = Path(source)
                image = Image.open(self.root / path if self.root and not path.is_absolute() else path)
            return self.preprocess(image.convert("RGB")), int(label), idx
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            return None, int(label), idx, repr(e)


def _collate(batch):
    kept = [b for b in batch if b[0] is not None]
    skipped = [(b[2], b[3]) for b in batch if b[0] is None]
    if not kept:
        return None, None, None, skipped
    return torch.stack([b[0] for b in kept]), torch.tensor([b[1] for b in kept]), torch.tensor([b[2] for b in kept]), skipped


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def encode_prompts(encoder, prompts: Sequence[str], batch_size: int = 256) -> torch.Tensor:
    """Normalised text embeddings aligned with ``prompts``; identical strings are encoded once."""
    unique = list(dict.fromkeys(prompts))
    chunks = [encoder.encode_text(unique[i : i + batch_size]).float().cpu() for i in range(0, len(unique), batch_size)]
    emb = l2_normalize(torch.cat(chunks))
    index = {p: i for i, p in enumerate(unique)}
    return emb[[index[p] for p in prompts]]


@torch.no_grad()
def encode_images(encoder, dataset: ImageRecords, batch_size: int = 32, num_workers: int = 0):
    """Returns ``(embeddings, labels, record_indices, skipped)`` with rows aligned."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    embs, labels, indices, skipped = [], [], [], []
    for images, batch_labels, batch_idx, batch_skipped in loader:
        skipped.extend(batch_skipped)
        if images is None:
            continue
        embs.append(l2_normalize(encoder.encode_image(images).float().cpu()))
        labels.append(batch_labels)
        indices.append(batch_idx)
    if not embs:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), skipped
    return torch.cat(embs), torch.cat(labels), torch.cat(indices), skipped


def cosine_scores(image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix ``[n_images, n_species]`` (inputs already normalised)."""
    return image_emb @ text_emb.T


@torch.no_grad()
def classify_images(encoder, images: Sequence[Image.Image], table: SpeciesTable, style: str, prefix: str = "", top_k: int = 5):
    """Rank every species for a few PIL images; returns, per image, a list of ``(Species, cosine)`` best first."""
    text_emb = encode_prompts(encoder, table.prompts(style, prefix))
    tensors = torch.stack([encoder.preprocess(im.convert("RGB")) for im in images])
    image_emb = l2_normalize(encoder.encode_image(tensors).float().cpu())
    top = cosine_scores(image_emb, text_emb).topk(min(top_k, len(table)), dim=1)
    return [[(table[j], float(v)) for v, j in zip(vals, idxs)] for vals, idxs in zip(top.values.tolist(), top.indices.tolist())]


@torch.no_grad()
def evaluate_zero_shot(
    encoder,
    records: Sequence[tuple],
    table: SpeciesTable,
    style: str,
    prefix: str = "",
    batch_size: int = 32,
    num_workers: int = 0,
    root: str | Path | None = None,
) -> dict:
    """Top-1 accuracy of ``encoder`` on ``records`` using one prompt per species."""
    t0 = time.time()
    prompts = table.prompts(style, prefix)
    dup = duplicate_prompt_groups(prompts)
    text_emb = encode_prompts(encoder, prompts)
    image_emb, labels, indices, skipped = encode_images(encoder, ImageRecords(records, encoder.preprocess, root), batch_size, num_workers)
    scores = cosine_scores(image_emb, text_emb) if image_emb.numel() else torch.empty(0, len(prompts))
    return {
        "prompt_style": style,
        "prompt_prefix": prefix,
        "prompt_example": prompts[0] if prompts else None,
        "n_species": len(table),
        "n_unique_prompts": len(set(prompts)),
        "duplicate_prompt_groups": {p: [table[i].scientific for i in ids] for p, ids in dup.items()},
        "n_records": len(records),
        "n_evaluated": int(labels.numel()),
        "n_skipped": len(skipped),
        "skipped": [{"record_index": int(i), "error": e} for i, e in skipped[:50]],
        "top1_accuracy_first": accuracy(scores, labels, "first"),
        "top1_accuracy_any": accuracy(scores, labels, "any"),
        "top5_accuracy": topk_accuracy(scores, labels, 5) if labels.numel() else float("nan"),
        "encoder": encoder.describe() if hasattr(encoder, "describe") else str(type(encoder).__name__),
        "seconds": time.time() - t0,
    }
