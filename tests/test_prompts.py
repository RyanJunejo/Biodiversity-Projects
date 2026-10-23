"""Prompt formats must reproduce the strings used in the 2024 notebooks."""

import pytest

from biodiv.clip_eval import Species
from biodiv.prompts import make_prompt

FEN = Species(0, "Dolomedes plantarius", "Fen raft spider",
              {"kingdom": "Animalia", "phylum": "Arthropoda", "class": "Arachnida", "order": "Araneae",
               "family": "Pisauridae", "genus": "Dolomedes", "species": "plantarius"})


def test_historical_formats():
    # rarespecies-common-taxi.ipynb cell 12
    assert make_prompt(FEN, "taxonomic_common", "a photo of ") == \
        "a photo of Animalia Arthropoda Arachnida Araneae Pisauridae Dolomedes plantarius with common name Fen raft spider"
    # rarespecies-common-taxi-nointro.ipynb cell 12 / BioClip-INAT21-Fixed (2).ipynb cell 8
    assert make_prompt(FEN, "taxonomic_common") == \
        "Animalia Arthropoda Arachnida Araneae Pisauridae Dolomedes plantarius with common name Fen raft spider"
    # rarespecies-common.ipynb cell 18
    assert make_prompt(FEN, "common", "a photo of ") == "a photo of Fen raft spider"
    # openclippy2 (1).ipynb cell 0
    assert make_prompt(FEN, "scientific", "a photo of a ") == "a photo of a Dolomedes plantarius"
    assert make_prompt(FEN, "scientific_common") == "Dolomedes plantarius with common name Fen raft spider"
    assert make_prompt(FEN, "taxonomic") == "Animalia Arthropoda Arachnida Araneae Pisauridae Dolomedes plantarius"


def test_unknown_style_and_missing_common_name():
    with pytest.raises(ValueError):
        make_prompt(FEN, "latin")
    with pytest.raises(ValueError):
        make_prompt(Species(0, "X y", None, FEN.taxonomy), "common")
