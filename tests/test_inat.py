"""Category mapping, record alignment, and the dataset honouring its transform."""


import pytest
import torch
from PIL import Image

from biodiv.fixtures import INAT_CATEGORIES
from biodiv.inat import CategoryMap, INatDataset, build_records, load_json, load_split


def test_category_map_handles_shuffled_noncontiguous_ids(inat_fixture):
    cats = load_json(inat_fixture["categories"])
    cmap = CategoryMap.from_categories(cats)
    assert cmap.category_ids == (3, 7, 42)  # sorted, not JSON order
    assert not cmap.is_contiguous()
    assert cmap.to_index(42) == 2 and cmap.to_category_id(2) == 42
    assert cmap.names[cmap.to_index(7)] == "Cicindela campestris"
    with pytest.raises(KeyError):
        cmap.to_index(99)


def test_duplicate_category_ids_rejected():
    with pytest.raises(ValueError):
        CategoryMap.from_categories([{"id": 1, "name": "a"}, {"id": 1, "name": "b"}])


def test_train_and_val_maps_agree_and_mismatch_is_detected(inat_fixture):
    _, train_map = load_split(inat_fixture["train"], inat_fixture["categories"])
    _, val_map = load_split(inat_fixture["val"], inat_fixture["categories"])
    train_map.assert_same(val_map)
    with pytest.raises(ValueError):
        train_map.assert_same(CategoryMap.from_categories([{"id": 3}, {"id": 7}]))


def test_records_pair_each_annotation_with_its_own_image(inat_fixture):
    data, cmap = load_split(inat_fixture["val"], inat_fixture["categories"])
    records = build_records(data, cmap)
    assert len(records) == len(data["annotations"])
    colour_by_name = {name: colour for _, name, colour in INAT_CATEGORIES}
    for file_name, label in records:
        # the directory in the path is the category id the image was generated for
        cid = int(file_name.split("/")[2])
        assert label == cmap.to_index(cid)
        pixel = Image.open(inat_fixture["image_root"] / file_name).getpixel((20, 20))
        assert pixel == colour_by_name[cmap.names[label]]


def test_unknown_image_id_raises(inat_fixture):
    data, cmap = load_split(inat_fixture["val"], inat_fixture["categories"])
    data["annotations"][0]["image_id"] = -1
    with pytest.raises(KeyError):
        build_records(data, cmap)


def test_dataset_uses_the_transform_it_was_given(inat_fixture):
    data, cmap = load_split(inat_fixture["val"], inat_fixture["categories"])
    records = build_records(data, cmap)
    sentinel = torch.full((3, 4, 4), 7.0)
    calls = []

    def transform(img):
        calls.append(img.size)
        return sentinel

    ds = INatDataset(records, inat_fixture["image_root"], transform=transform)
    image, label = ds[0]
    assert torch.equal(image, sentinel) and calls == [(32, 32)]
    assert label == records[0][1]
    raw, _ = INatDataset(records, inat_fixture["image_root"], transform=None)[0]
    assert isinstance(raw, Image.Image)


def test_category_lookup_map_is_built_once():
    cmap = CategoryMap.from_categories([{"id": i, "name": str(i)} for i in range(1000)])
    assert cmap._id_to_index is cmap._id_to_index  # cached, not rebuilt per lookup
    assert cmap.to_index(999) == 999
