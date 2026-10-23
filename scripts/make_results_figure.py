#!/usr/bin/env python
"""Render docs/figures/results_top1.png from results/results_inventory.json.

One measure (top-1 accuracy), grouped by evaluation dataset. Two kinds of bar:
values printed by a 2024 notebook (or read from a W&B chart, shown as a range)
and values that exist only on a slide. Nothing here was re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "results_inventory.json"
OUT = REPO / "docs" / "figures" / "results_top1.png"

# Validated categorical slots 1 and 2 (dataviz reference palette) + chrome ink.
RECORDED, SLIDE_ONLY = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

# (inventory id, label) in display order, grouped by evaluation set.
ROWS = [
    ("iNaturalist 2019 val (3,030 images, 1,010 species)", [
        ("H1", "Hiera-tiny, fine-tuned (val images were augmented)"),
        ("B1", "BioCLIP zero-shot, \"a photo of a {scientific name}\""),
        ("B11", "OpenCLIP zero-shot (prompt unknown)"),
        ("B10", "OpenAI CLIP zero-shot (prompt unknown)"),
    ]),
    ("iNaturalist 2021 val (100,000 images, 10,000 species)", [
        ("H2", "Hiera-tiny, fine-tuned on train_mini (val images were augmented)"),
        ("B3", "BioCLIP iNat21-only, taxonomy + common name (text unnormalised)"),
        ("B2", "BioCLIP, taxonomy + common name (text unnormalised)"),
    ]),
    ("Rare Species (11,983 images, 400 species; unseen by BioCLIP)", [
        ("B6", "BioCLIP, taxonomy + common name"),
        ("B5", "BioCLIP, \"a photo of\" + taxonomy + common name"),
        ("B9", "BioCLIP, scientific + common name"),
        ("B8", "BioCLIP, taxonomy only"),
        ("B7", "BioCLIP, scientific name only"),
        ("B4", "BioCLIP, \"a photo of\" + common name (398-way: 2 shared names)"),
    ]),
]


def main() -> None:
    entries = {r["id"]: r for r in json.loads(INVENTORY.read_text())["results"]}
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y, ticks, labels = 0.0, [], []
    for group, items in ROWS:
        ax.text(-0.015, y + 0.55, group, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK)
        y += 1.0
        for rid, label in items:
            e = entries[rid]
            slide_only = e["status"] == "slide only, unverified"
            colour = SLIDE_ONLY if slide_only else RECORDED
            if "value_range" in e:  # Hiera: a range read from the W&B chart
                lo, hi = e["value_range"]
                ax.barh(y, lo, height=0.62, color=colour, edgecolor="none")
                ax.barh(y, hi - lo, left=lo, height=0.62, color=colour, alpha=0.35, edgecolor="none")
                ax.text(hi + 0.012, y, f"{lo:.2f}-{hi:.2f} (chart)", va="center", fontsize=8.5, color=INK)
            else:
                ax.barh(y, e["value"], height=0.62, color=colour, edgecolor="none")
                text = f"{e['value']:.3f}" if e["value_precision"] == "exact" else f"{e['value']:.2f}"
                ax.text(e["value"] + 0.012, y, text, va="center", fontsize=8.5, color=INK)
            ticks.append(y)
            labels.append(label)
            y += 1.0
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5, color=MUTED)
    ax.set_xlabel("top-1 accuracy", fontsize=9, color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color="#c3c2b7")
    ax.legend(handles=[Patch(color=RECORDED, label="printed by a 2024 notebook, or read from a W&B chart (range)"),
                       Patch(color=SLIDE_ONLY, label="appears on a 2024 slide only; no notebook or log")],
              loc="upper left", bbox_to_anchor=(0.0, -0.075), fontsize=8.5, frameon=False, ncol=1)
    fig.text(0.01, 0.975, "Top-1 accuracy as recorded in 2024 (no experiment re-run)", fontsize=11.5, color=INK, fontweight="bold", va="top")
    fig.text(0.01, 0.012, "Hiera = supervised fine-tuning. BioCLIP = zero-shot prompts; every iNaturalist species here was in its pre-training data,\n"
             "so Rare Species is the only unseen-species test. Per-bar caveats: results/results.md.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
