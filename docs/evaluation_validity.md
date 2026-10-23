# Evaluation validity: what each checkpoint had seen before it was evaluated

Each row is one checkpoint / dataset pairing that appears in the 2024 results. "Confirmed"
means stated by the upstream dataset or model documentation (links in `README.md`); nothing
below was independently verified by comparing image files. **Absence of overlap evidence is not
evidence of disjointness**; the "Unknown" column lists what is still open.

## Summary table

| Checkpoint | Pre-training data | Evaluation data (split) | Fine-tuned? | Image overlap with training | Species overlap with training | Unknown |
|---|---|---|---|---|---|---|
| Hiera-tiny, `mae_in1k` (MAE self-supervised on ImageNet-1k, random head) | ImageNet-1k images (no labels) | iNat 2019 **val** (3,030 images, 1,010 species) after fine-tuning on iNat 2019 train (265,213) | Yes, full fine-tuning | Train/val are disjoint by the dataset's construction (confirmed by the competition README's split files). ImageNet-1k vs iNaturalist: not checked. | All 1,010 species are in the fine-tuning set: evaluation on **unseen images of seen species**. | Whether ImageNet-1k contains any of the same iNaturalist photos; which epoch the reported peak refers to; the trained weights (not in the repository). |
| Hiera-tiny, `mae_in1k` | ImageNet-1k (no labels) | iNat 2021 **val** (100,000 images, 10/species) after fine-tuning on iNat 2021 **train_mini** (500,000, 50/species) | Yes | Train/val disjoint by construction (confirmed). | All 10,000 species seen: unseen images of seen species. | Same as above. |
| BioCLIP `imageomics/bioclip` (ViT-B/16, from OpenAI CLIP weights) | TreeOfLife-10M: iNat21 **training split** (2.7M) + BIOSCAN-1M + EOL (6.6M); ~454K taxa | iNat 2021 **val** (100,000) | No (zero-shot prompts) | Card states only the iNat21 *training* split was ingested, so val images should be absent (confirmed by statement, not by file comparison). | **All 10,000 species seen** in pre-training, with iNaturalist's own taxonomic and common-name text. This is zero-shot only in the sense of no fine-tuning; it is not an unseen-species evaluation. | Whether any iNat21 val photo also appears in EOL; the exact Hub revision used in 2024. |
| BioCLIP `imageomics/bioclip-vit-b-16-inat-only` | iNat21 training split only (~5% held out as its own validation) | iNat 2021 val | No | Card: trained on iNat21; its internal validation set was carved from the training samples, so the official val split should be unseen (confirmed by statement). | All 10,000 species seen. | Whether the ~5% internal validation subset overlaps the official val split (card does not say). |
| BioCLIP `imageomics/bioclip` | TreeOfLife-10M (as above) | iNat 2019 **val** (3,030 images, 1,010 species) | No | **Unknown.** iNat 2019 and iNat 2021 were both drawn from iNaturalist observations; the 2021 README says "there is certainly overlap between the species and images included in the various datasets". TreeOfLife-10M includes iNat21 train, so iNat 2019 val photos may be in BioCLIP's training data. | Unknown but likely high: the 2019 species were selected from genera with at least 10 species and most are plausibly among the 10,000 iNat21 species; no mapping is provided upstream. | Image and species overlap, both unquantified. |
| BioCLIP `imageomics/bioclip` | TreeOfLife-10M **minus** the 400 Rare Species species | Rare Species (11,983 EOL images, 400 species) | No | Card: the 400 species (and so their images) were removed before training (confirmed by statement). | **None by construction**: unseen species. This is the only genuinely unseen-species evaluation in the repository. Note the card's caveat that 5 species have duplicate images within the set. | Whether EOL images of the same *individual photos* exist under other taxon pages in TreeOfLife-10M; the Hub revision used in 2024. |

## Reading the table

* **Evaluation without fine-tuning** (all BioCLIP rows) means the weights were not updated; it
  says nothing about whether the species were seen. For the iNat21 rows they all were.
* **Evaluation on unseen data** in the strong sense (species absent from every training stage)
  holds only for the Rare Species row.
* The Hiera rows are ordinary supervised train/validation evaluations, which is the standard
  setting for those competitions; they are not comparable to the BioCLIP zero-shot rows.

## TreeOfLife-10M and iNaturalist, specifically

From the TreeOfLife-10M dataset card (Hugging Face, revision current on 2026-10-23):

* "iNat21 data was downloaded, unzipped, and our compilation scripts pointed to the training
  split"; iNat21 images are catalogued (by `inat21_filename`, `inat21_cls_num`) but not
  redistributed.
* iNaturalist English vernacular names and taxa were used as the text for those images, which
  is why the `{taxonomy} with common name {common}` prompt format matches BioCLIP's training
  text for iNat21 species.
* "Species selected for the Rare Species dataset were removed from this dataset."
* The `bioclip-vit-b-16-inat-only` card: "iNat21 is a subset of the TreeOfLife-10M dataset on
  which BioCLIP was trained."

## Confirmed facts vs open questions

Confirmed (by upstream documentation): TreeOfLife-10M ⊇ iNat21 train; Rare Species species ∉
TreeOfLife-10M; iNat val splits are disjoint from their train splits; Hiera `mae_in1k` is
self-supervised on ImageNet-1k.

Open: iNat 2019 ↔ iNat21/TreeOfLife-10M image and species overlap; ImageNet-1k ↔ iNaturalist
image overlap; the 2024 Hub revisions of the BioCLIP checkpoints (both repositories have been
modified since: `imageomics/bioclip` last modified 2026-02-26, `bioclip-vit-b-16-inat-only`
2025-11-18; whether the weight files changed is not recorded here); the open_clip version used
in 2024 and therefore the exact preprocessing applied.
