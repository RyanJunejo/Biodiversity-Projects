# Results summary (evidence-backed)

Every number below was read from an artifact in this repository; none was re-run in 2026.
The machine-readable version is `results/results_inventory.json`. "Status" values:

* **historical recorded output**: printed by a 2024 notebook cell whose output is preserved
  verbatim in `archive/2024-original/notebooks/`, or logged in the W&B report PDF.
* **historical approximate**: only a chart or rounded slide value exists; the exact number is
  not recoverable.
* **slide only, unverified**: appears on a slide with no notebook, log or run artifact.
* **newly reproduced**: none. No experiment was re-run; see `docs/reproduction.md` for how to.

Datasets and checkpoints are described in `docs/evaluation_validity.md`; where the current code
differs from the scripts that produced the historical numbers is described in `docs/changes.md`.

## Hiera-tiny fine-tuning (supervised)

| # | Model / checkpoint | Dataset (split) | Images / classes | Metric | Value | Source | Known issues | Status |
|---|---|---|---|---|---|---|---|---|
| H1 | Hiera-tiny 224, init `mae_in1k` (MAE, ImageNet-1k), fine-tuned on iNat 2019 train | iNat 2019 val | 3,030 / 1,010 | top-1 val accuracy, best epoch | **~0.67 (chart) to 0.69 (report text); slide says "peaks at 0.7"** | W&B report PDF p.3 chart; slide 13 (same chart) | Val images were passed through the *training* augmentation (random flip + RandAugment). Run logged ~45 of the configured 300 epochs at report time. Drop path 0.0 (not 0.1), MixUp alpha 1.0 (not 0.8). Recorded val loss invalid (softmax applied twice). No checkpoint or metrics file survives. | historical approximate |
| H2 | Hiera-tiny 224, init `mae_in1k`, fine-tuned on iNat 2021 train_mini | iNat 2021 val | 100,000 / 10,000 | top-1 val accuracy, best epoch | **~0.61** (peak around epoch 25-30, declining to ~0.56 by epoch ~52) | W&B report PDF p.1-2; slide 14 | Same as H1. Training images: 500,000 (50/species), not "10 per species" as the report text says. | historical approximate |

Training setup common to H1/H2 (from the scripts and the W&B report): batch 512, AdamW
(0.9, 0.999), weight decay 0.05, label smoothing 0.1, CutMix/MixUp (alpha 1.0 each in
practice), RandAugment 9 ops magnitude 1, layer-wise LR decay 0.65 from 1.95e-3 at the head,
constant 0.1x LR for 5 epochs then cosine (T_max 300), fp16 autocast, one NVIDIA A40,
~30 min/epoch.

## BioCLIP zero-shot classification (no fine-tuning)

| # | Checkpoint | Dataset (split) | Images / classes | Prompt format | Metric | Value | Source | Known issues | Status |
|---|---|---|---|---|---|---|---|---|---|
| B1 | `hf-hub:imageomics/bioclip` (revision not recorded) | iNat 2019 val | 3,030 / 1,010 | `a photo of a {scientific name}` | top-1 | **0.7063** | `archive/2024-original/notebooks/openclippy2 (1).ipynb` cell 11 (also `openclippy2 (with results).ipynb`, `BioClip-INAT21.ipynb` cell 16) | Notebook named "openclippy" but uses BioCLIP. Prediction index assumed equal to category id (true for iNat 2019, implicit). fp16 autocast. | historical recorded output |
| B2 | `hf-hub:imageomics/bioclip` | iNat 2021 val | 100,000 / 10,000 | `{kingdom ... genus epithet} with common name {common}` (no prefix) | top-1 | **0.6398** | `BioClip-INAT21-Fixed (2).ipynb` cell 21 | **Text embeddings not L2-normalised** (image embeddings were), so scores were cosine x text-norm, not the cosine similarity used upstream. Label index assumed equal to category id. All 10,000 species present in BioCLIP's pre-training data. | historical recorded output |
| B3 | `hf-hub:imageomics/bioclip-vit-b-16-inat-only` | iNat 2021 val | 100,000 / 10,000 | same as B2 | top-1 | **0.7195** (slide 20 rounds to 0.72) | `BioClip-INAT21-Fixed (2).ipynb` cell 25 | As B2, plus the images were preprocessed with the *first* checkpoint's `preprocess` object rather than this checkpoint's. | historical recorded output |
| B4 | `hf-hub:imageomics/bioclip` | Rare Species (single `train` split) | 11,983 / **398** labels | `a photo of {common name}` | top-1 | **0.2944** (3,528 / 11,983) | `rarespecies-common.ipynb` cells 28-31 | Only 398 unique common names: *Acropora acuminata* + *A. millepora* ("staghorn coral") and *Tylototriton shanjing* + *T. verrucosus* ("Yunnan Newt") were merged into one label each, so this is a 398-way task scored leniently for those four species. Not the same task as B5/B6. | historical recorded output |
| B5 | `hf-hub:imageomics/bioclip` | Rare Species | 11,983 / 400 | `a photo of {taxonomy} with common name {common}` | top-1 | **0.3717** | `rarespecies-common-taxi.ipynb` cell 29 | Label list built with `range(len(dataset) - 2)`; the two skipped rows belonged to species already listed, so no effect. One image triggers Pillow's decompression-bomb warning (still processed). | historical recorded output |
| B6 | `hf-hub:imageomics/bioclip` | Rare Species | 11,983 / 400 | `{taxonomy} with common name {common}` | top-1 | **0.3739** | `rarespecies-common-taxi-nointro.ipynb` cell 29 | As B5. | historical recorded output |
| B7 | `hf-hub:imageomics/bioclip` | Rare Species | 11,983 / ? | scientific name | top-1 | 0.313 | slide 19 chart data only | No notebook, no label count. | slide only, unverified |
| B8 | `hf-hub:imageomics/bioclip` | Rare Species | 11,983 / ? | taxonomic name | top-1 | 0.327 | slide 19 | No notebook. | slide only, unverified |
| B9 | `hf-hub:imageomics/bioclip` | Rare Species | 11,983 / ? | scientific + common | top-1 | 0.36 | slide 19 | No notebook. | slide only, unverified |
| B10 | OpenAI CLIP (ViT variant unknown) | iNat 2019 (val presumed) | ? | unknown | top-1 | 0.06 | slide 17 chart data | No notebook, prompt or model size recorded. | slide only, unverified |
| B11 | OpenCLIP (LAION; variant unknown) | iNat 2019 (val presumed) | ? | unknown | top-1 | 0.08 | slide 17 | No notebook. | slide only, unverified |

For context only (not this project's results): the Rare Species dataset card reports BioCLIP
zero-shot top-1 of 38.09% with taxonomic + common prompts, OpenAI CLIP 31.81% and OpenCLIP
29.85% with common-name prompts, and the iNat21-only model 21.33%.

## On the three rare-species values (29.44%, 37.17%, 37.39%)

* Provenance: each is the printed `Accuracy:` line of the final evaluation cell in its
  notebook; the common-name notebook additionally prints `correct = 3528` and
  `len(image_labels) = 11983`, and 3528 / 11983 = 0.29442, matching.
* They are **not on one task**. B4 scores 398 classes; B5 and B6 score 400. For the four
  species involved, B4 counts a prediction of the shared name as correct for both species,
  which can only raise its accuracy relative to a strict 400-way scoring. The 7.7-point gap
  between B4 and B5/B6 is much larger than the at-most 0.5 points (60 images) the merge could
  account for, so the ordering is robust, but the exact values are not directly comparable.
* B5 vs B6 differ by 0.22 points (26 images). No seed, run-to-run or bootstrap variability was
  recorded, so whether the prefix matters cannot be concluded from these two numbers alone.
* The replacement notebook `notebooks/bioclip_rare_species_prompts.ipynb` scores all formats on
  the same 400 species with an explicit tie policy; it has not been run on the real data here.

## Uncertainty

No standard errors or confidence intervals are given because the per-image predictions were
not saved in 2024; only the aggregate accuracies survive. Recomputing them requires re-running
the evaluations (`scripts/eval_bioclip.py` writes per-run JSON including the counts needed).
