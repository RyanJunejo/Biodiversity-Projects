"""Adapters checked against *real* upstream record formats (not fixtures I wrote).

Samples are copied from recorded 2024 notebook outputs and from the pinned
Hugging Face dataset card, so a change in an upstream schema shows up here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from biodiv.clip_eval import ImageRecords, SpeciesTable
from biodiv.species_data import inat_split, rare_species_from_dataset

REPO = Path(__file__).resolve().parents[1]

# archive/2024-original/notebooks/How to Navigate iNaturalist (1).ipynb, cell 21 (iNat 2019 categories.json)
INAT2019_CATEGORY = {"kingdom": "Fungi", "phylum": "Basidiomycota", "name": "Amanita vaginata", "family": "Amanitaceae",
                     "genus": "Amanita", "order": "Agaricales", "id": 0, "class": "Agaricomycetes"}
# archive/2024-original/notebooks/BioClip-INAT21-Fixed (2).ipynb, cell 6 (iNat 2021 val.json categories)
INAT2021_CATEGORY = {"id": 0, "name": "Lumbricus terrestris", "common_name": "Common Earthworm", "supercategory": "Animalia",
                     "kingdom": "Animalia", "phylum": "Annelida", "class": "Clitellata", "order": "Haplotaxida",
                     "family": "Lumbricidae", "genus": "Lumbricus", "specific_epithet": "terrestris",
                     "image_dir_name": "00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris"}
# huggingface.co/datasets/imageomics/rare-species @ 7e984e1f: features list (image column is ``file_name``)
RARE_SPECIES_COLUMNS_2026 = ["file_name", "rarespecies_id", "eol_content_id", "eol_page_id", "kingdom", "phylum", "class",
                             "order", "family", "genus", "species", "sciName", "common"]
RARE_SPECIES_COLUMNS_2024 = ["image"] + RARE_SPECIES_COLUMNS_2026[1:]


def test_inat2019_categories_without_epithet_or_common_name():
    table = SpeciesTable.from_inat_categories([INAT2019_CATEGORY])
    s = table[0]
    assert s.scientific == "Amanita vaginata" and s.taxonomy["species"] == "vaginata" and s.common is None
    assert table.prompts("scientific", "a photo of a ") == ["a photo of a Amanita vaginata"]  # the 2024 iNat 2019 prompt
    assert table.prompts("taxonomic") == ["Fungi Basidiomycota Agaricomycetes Agaricales Amanitaceae Amanita vaginata"]
    with pytest.raises(ValueError):
        table.prompts("common")  # iNat 2019 has no common names: must fail loudly, not emit "None"


def test_inat2021_categories_use_the_epithet_and_common_name():
    table = SpeciesTable.from_inat_categories([INAT2021_CATEGORY])
    assert table.prompts("taxonomic_common") == [
        "Animalia Annelida Clitellata Haplotaxida Lumbricidae Lumbricus terrestris with common name Common Earthworm"]


def test_inat_split_with_separate_categories_file_writes_nothing(tmp_path):
    import json

    (tmp_path / "val2019.json").write_text(json.dumps({
        "images": [{"id": 7, "file_name": "train_val2019/Fungi/0/x.jpg"}],
        "annotations": [{"id": 7, "image_id": 7, "category_id": 0}],
        "categories": [{"id": 0, "name": "obfuscated-0"}],  # as shipped in the 2019 split files
    }))
    (tmp_path / "categories.json").write_text(json.dumps([INAT2019_CATEGORY]))
    listing = sorted(os.listdir(tmp_path))
    records, table = inat_split(tmp_path / "val2019.json", "/data/inat_comp/2019", tmp_path / "categories.json")
    assert table[0].scientific == "Amanita vaginata"
    assert records == [("/data/inat_comp/2019/train_val2019/Fungi/0/x.jpg", 0)]
    assert sorted(os.listdir(tmp_path)) == listing  # no temp file next to the dataset


class _FakeHubDataset:
    """Minimal stand-in for datasets.Dataset: column_names, remove_columns, iteration, indexing."""

    def __init__(self, columns, rows):
        self.column_names, self._rows, self.decodes = list(columns), rows, 0

    def remove_columns(self, names):
        keep = [c for c in self.column_names if c not in names]
        return _FakeHubDataset(keep, [{k: r[k] for k in keep} for r in self._rows])

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        self.decodes += 1
        return self._rows[i]


def _rare_rows(image_col):
    base = {"rarespecies_id": "x", "eol_content_id": 1, "eol_page_id": 2, "kingdom": "Animalia", "phylum": "Cnidaria",
            "class": "Anthozoa", "order": "Scleractinia", "family": "Acroporidae", "genus": "Acropora"}
    rows = []
    for sp in ("acuminata", "millepora"):
        rows.append({**base, image_col: Image.new("RGB", (8, 8), "red"), "species": sp, "sciName": f"Acropora {sp}", "common": "staghorn coral"})
    return rows


@pytest.mark.parametrize("columns", [RARE_SPECIES_COLUMNS_2026, RARE_SPECIES_COLUMNS_2024])
def test_rare_species_hub_schema_and_lazy_decoding(columns):
    ds = _FakeHubDataset(columns, _rare_rows(columns[0]))
    records, table = rare_species_from_dataset(ds)
    assert len(table) == 2 and [s.scientific for s in table] == ["Acropora acuminata", "Acropora millepora"]
    assert ds.decodes == 0  # nothing decoded at load time
    image, label, idx = ImageRecords(records, lambda im: im)[1]
    assert ds.decodes == 1 and isinstance(image, Image.Image) and label == 1 and idx == 1


def test_eval_cli_smoke_runs_without_repo_root_on_path(tmp_path):
    """The documented quick-start command must work with only the package importable."""
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, str(REPO / "scripts/eval_bioclip.py"), "--dataset", "fixture", "--checkpoint", "lookup",
                          "--prompt-style", "common", "--prefix", "a photo of "], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert '"top1_accuracy_any": 1.0' in out.stdout
