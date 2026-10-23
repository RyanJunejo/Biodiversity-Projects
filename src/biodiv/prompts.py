"""Text prompt formats for CLIP-style zero-shot species classification.

Styles follow Table 3 of the BioCLIP paper (Stevens et al., 2024):

    common              black-billed magpie
    scientific          Pica hudsonia
    taxonomic           Animalia Chordata Aves Passeriformes Corvidae Pica hudsonia
    scientific_common   Pica hudsonia with common name black-billed magpie
    taxonomic_common    Animalia Chordata Aves Passeriformes Corvidae Pica hudsonia with common name black-billed magpie

``prefix`` is prepended verbatim. The 2024 notebooks used ``"a photo of "``,
``"a photo of a "`` and ``""``; all three are reproducible here.
"""

from __future__ import annotations

from typing import Iterable

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
PROMPT_STYLES = ("common", "scientific", "taxonomic", "scientific_common", "taxonomic_common")


def make_prompt(species, style: str, prefix: str = "") -> str:
    """Build one prompt for a ``Species`` (see ``clip_eval``) or a plain dict."""
    get = species.get if isinstance(species, dict) else lambda k, d=None: getattr(species, k, d)
    taxonomy = get("taxonomy") or {r: get(r) for r in RANKS}
    scientific = get("scientific") or f"{taxonomy['genus']} {taxonomy['species']}"
    common = get("common")
    if style in ("taxonomic", "taxonomic_common"):
        missing = [r for r in RANKS if taxonomy.get(r) in (None, "")]
        if missing:
            raise ValueError(f"species {scientific!r} lacks taxonomy ranks {missing} needed for style {style!r}")
    taxonomic = " ".join(str(taxonomy[r]) for r in RANKS)

    if style == "common":
        if not common:
            raise ValueError(f"species {scientific!r} has no common name")
        text = str(common)
    elif style == "scientific":
        text = scientific
    elif style == "taxonomic":
        text = taxonomic
    elif style == "scientific_common":
        text = f"{scientific} with common name {common}"
    elif style == "taxonomic_common":
        text = f"{taxonomic} with common name {common}"
    else:
        raise ValueError(f"unknown prompt style {style!r}; choose from {PROMPT_STYLES}")
    return f"{prefix}{text}"


def make_prompts(species_list: Iterable, style: str, prefix: str = "") -> list[str]:
    return [make_prompt(s, style, prefix) for s in species_list]
