"""Species identity vs prompt text, alignment, normalisation, and hand-checked metrics."""

import math

import torch

from biodiv.clip_eval import ImageRecords, cosine_scores, duplicate_prompt_groups, encode_images, encode_prompts, evaluate_zero_shot
from biodiv.metrics import accuracy, top1_correct, topk_accuracy
from biodiv.species_data import inat21_val
from tests.fixtures.make_fixture import fixture_encoder, load_fixture_rare_species


def test_duplicate_common_names_do_not_merge_species():
    records, table = load_fixture_rare_species(images_per_species=2)
    assert len(table) == 4  # species identity is kept
    prompts = table.prompts("common", "a photo of ")
    assert len(set(prompts)) == 3  # ... even though only 3 prompts are unique
    assert duplicate_prompt_groups(prompts) == {"a photo of staghorn coral": [0, 1]}
    labels = sorted({label for _, label in records})
    assert labels == [0, 1, 2, 3]


def test_tie_policy_is_explicit_for_shared_prompts():
    records, table = load_fixture_rare_species(images_per_species=2)
    enc = fixture_encoder()
    common = evaluate_zero_shot(enc, records, table, "common", "a photo of ")
    scientific = evaluate_zero_shot(enc, records, table, "scientific")
    # identical prompt text -> identical scores -> one of the two corals always loses under "first"
    assert common["top1_accuracy_first"] == 0.75
    assert common["top1_accuracy_any"] == 1.0
    assert common["n_unique_prompts"] == 3 and common["duplicate_prompt_groups"]
    # distinct scientific names separate them again
    assert scientific["top1_accuracy_first"] == 1.0 and scientific["n_unique_prompts"] == 4


def test_predictions_stay_aligned_when_an_image_is_unreadable(tmp_path):
    records, table = load_fixture_rare_species(images_per_species=1)
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    # insert the broken record in the middle so a shift would move every later label
    records = records[:2] + [(str(bad), 2)] + records[2:]
    enc = fixture_encoder()
    emb, labels, indices, skipped = encode_images(enc, ImageRecords(records, enc.preprocess), batch_size=2)
    assert [i for i, _ in skipped] == [2] and len(labels) == 4
    assert indices.tolist() == [0, 1, 3, 4]
    assert labels.tolist() == [records[i][1] for i in indices.tolist()]
    result = evaluate_zero_shot(enc, records, table, "scientific")
    assert result["n_skipped"] == 1 and result["n_evaluated"] == 4 and result["top1_accuracy_first"] == 1.0


def test_scores_are_cosine_similarities_regardless_of_input_norms():
    class Scaled:  # returns badly scaled, unnormalised vectors
        preprocess = None

        def encode_text(self, texts):
            return torch.tensor([[3.0, 4.0], [0.0, 50.0]])

    text = encode_prompts(Scaled(), ["a", "b"])
    assert torch.allclose(text.norm(dim=1), torch.ones(2))
    image = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    scores = cosine_scores(image, text)
    assert math.isclose(scores[0, 0].item(), 0.6, abs_tol=1e-6)  # cos((1,0),(3,4)) = 3/5
    assert math.isclose(scores[0, 1].item(), 0.0, abs_tol=1e-6)


def test_identical_prompts_get_identical_embeddings():
    enc = fixture_encoder()
    emb = encode_prompts(enc, ["staghorn coral", "Fen raft spider", "staghorn coral"])
    assert torch.equal(emb[0], emb[2]) and not torch.equal(emb[0], emb[1])


def test_metrics_match_hand_calculation():
    scores = torch.tensor([
        [0.9, 0.1, 0.0],  # label 0 -> correct
        [0.2, 0.7, 0.1],  # label 0 -> wrong at top-1, correct at top-2
        [0.5, 0.5, 0.0],  # label 1 -> exact tie with class 0
        [0.0, 0.3, 0.6],  # label 2 -> correct
    ])
    labels = torch.tensor([0, 0, 1, 2])
    assert top1_correct(scores, labels, "first").tolist() == [True, False, False, True]
    assert top1_correct(scores, labels, "any").tolist() == [True, False, True, True]
    assert accuracy(scores, labels, "first") == 0.5 and accuracy(scores, labels, "any") == 0.75
    assert topk_accuracy(scores, labels, 2) == 1.0


def test_inat21_labels_use_explicit_category_mapping(tmp_path):
    categories = [
        {"id": 5000, "name": "Pica hudsonia", "common_name": "Black-billed Magpie", "kingdom": "Animalia", "phylum": "Chordata",
         "class": "Aves", "order": "Passeriformes", "family": "Corvidae", "genus": "Pica", "specific_epithet": "hudsonia"},
        {"id": 12, "name": "Lumbricus terrestris", "common_name": "Common Earthworm", "kingdom": "Animalia", "phylum": "Annelida",
         "class": "Clitellata", "order": "Haplotaxida", "family": "Lumbricidae", "genus": "Lumbricus", "specific_epithet": "terrestris"},
    ]
    data = {"images": [{"id": 1, "file_name": "val/a.jpg"}, {"id": 2, "file_name": "val/b.jpg"}],
            "annotations": [{"image_id": 2, "category_id": 5000}, {"image_id": 1, "category_id": 12}],
            "categories": categories}
    p = tmp_path / "val.json"
    p.write_text(__import__("json").dumps(data))
    records, table = inat21_val(p, "/root")
    assert [s.scientific for s in table] == ["Lumbricus terrestris", "Pica hudsonia"]  # ordered by id
    assert records == [("/root/val/b.jpg", 1), ("/root/val/a.jpg", 0)]
    assert table.prompts("taxonomic_common")[1] == \
        "Animalia Chordata Aves Passeriformes Corvidae Pica hudsonia with common name Black-billed Magpie"
