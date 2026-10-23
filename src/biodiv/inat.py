"""iNaturalist competition annotation handling (2019 and 2021 JSON formats).

Both years ship COCO-style JSON::

    {"images": [{"id": ..., "file_name": ...}, ...],
     "annotations": [{"image_id": ..., "category_id": ...}, ...],
     "categories": [{"id": ..., "name": ..., ...}, ...]}

Category ids happen to be contiguous 0..N-1 in both years, but nothing in the
code below relies on that: ids are mapped to class indices explicitly, and the
training and validation mappings are checked against each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class CategoryMap:
    """Explicit two-way mapping between dataset category ids and class indices.

    Class index i corresponds to ``category_ids[i]``; ids are sorted so the
    mapping is deterministic regardless of the order in the JSON file.
    """

    category_ids: tuple[int, ...]
    names: tuple[str, ...]

    @classmethod
    def from_categories(cls, categories: Sequence[dict]) -> "CategoryMap":
        cats = sorted(categories, key=lambda c: int(c["id"]))
        ids = tuple(int(c["id"]) for c in cats)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate category ids in categories list")
        names = tuple(str(c.get("name", c["id"])) for c in cats)
        return cls(ids, names)

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    @property
    def _id_to_index(self) -> dict[int, int]:
        return {cid: i for i, cid in enumerate(self.category_ids)}

    def to_index(self, category_id: int) -> int:
        try:
            return self._id_to_index[int(category_id)]
        except KeyError as e:
            raise KeyError(f"category id {category_id} not in category map") from e

    def to_category_id(self, index: int) -> int:
        return self.category_ids[index]

    def is_contiguous(self) -> bool:
        return self.category_ids == tuple(range(self.num_classes))

    def assert_same(self, other: "CategoryMap") -> None:
        if self.category_ids != other.category_ids:
            raise ValueError(
                "category maps differ: "
                f"{self.num_classes} vs {other.num_classes} classes, "
                f"first mismatch at index "
                f"{next((i for i, (a, b) in enumerate(zip(self.category_ids, other.category_ids)) if a != b), min(self.num_classes, other.num_classes))}"
            )


def build_records(data: dict, category_map: CategoryMap) -> list[tuple[str, int]]:
    """Return ``(file_name, class_index)`` pairs, one per annotation.

    Each annotation is paired with its image through ``image_id``; an
    annotation that points at an unknown image raises instead of being
    silently dropped, so images and labels can never go out of step.
    """
    images = {img["id"]: img["file_name"] for img in data["images"]}
    records = []
    for ann in data["annotations"]:
        file_name = images.get(ann["image_id"])
        if file_name is None:
            raise KeyError(f"annotation {ann.get('id')} references unknown image id {ann['image_id']}")
        records.append((file_name, category_map.to_index(ann["category_id"])))
    return records


class INatDataset(Dataset):
    """Image-classification dataset over ``(file_name, class_index)`` records.

    The transform passed to the constructor is the one that is applied (the
    original 2024 class ignored its argument and used a module-level global).
    """

    def __init__(
        self,
        records: Sequence[tuple[str, int]],
        root: str | Path,
        transform: Callable | None = None,
    ):
        self.records = list(records)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        file_name, label = self.records[idx]
        image = Image.open(self.root / file_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_split(json_path: str | Path, categories_json: str | Path | None = None):
    """Load an annotation file and return ``(data, CategoryMap)``.

    ``categories_json`` optionally overrides the ``categories`` list (iNat 2019
    ships obfuscated names inside the split files and real names separately).
    """
    data = load_json(json_path)
    categories = load_json(categories_json) if categories_json else data["categories"]
    return data, CategoryMap.from_categories(categories)
