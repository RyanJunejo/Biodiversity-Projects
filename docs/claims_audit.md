# Audit of the research claims in the 2024 materials

Scope: every script, notebook, README, presentation and saved result in the repository as of
tag `original-2024`, checked against the code itself, the recorded notebook outputs, the
Weights & Biases report PDF, the slide decks, and the upstream documentation of the datasets
and models (iNaturalist 2019/2021 READMEs, the TreeOfLife-10M / Rare Species / BioCLIP cards on
the Hugging Face Hub, the Hiera source). Verification was done in October 2026 without
re-running any GPU experiment.

Legend: **Supported** = artifact in the repository backs the claim. **Partly** = backed with
caveats. **Unsupported** = no artifact; the claim was removed or relabelled in the new README.
**Incorrect** = contradicted by code or upstream documentation.

## Models

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 1 | "Hiera is a Vision Transformer model specialized in hierarchical classification, which aligns well with taxonomic classification" (README) | **Incorrect** | Hiera's hierarchy is spatial: it pools tokens across four stages of decreasing resolution (Ryali et al., 2023). Both training scripts fine-tune a flat `nn.Linear` head with plain cross-entropy over 1,010 or 10,000 species; no taxonomic rank, hierarchy loss or hierarchical label is used anywhere. Slide 8 of the deck describes Hiera correctly; the README over-interpreted the word "hierarchical". |
| 2 | "Hiera outperforms standard ViTs in handling hierarchical data structures" / "Other models, such as standard ViTs, are also tested" (README) | **Unsupported** | No ViT baseline was trained or evaluated in any artifact. Slide 14 compares the iNat21-mini peak (0.61) to a published WaveMix-256/16 number, which is a citation, not an experiment, and WaveMix is not a ViT. The W&B report's "matches the top visual transformer trained and evaluated on the dataset" cites nothing. |
| 3 | "BioCLIP ... improving its performance in distinguishing similar or visually ambiguous species" (README) | Partly | Paraphrase of the BioCLIP paper's claims; nothing in this repository measures fine-grained confusions. Credited to the upstream paper in the new README. |

## Methods and metrics

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 4 | "Two key metrics are introduced: a precision-recall based metric ... and an embedding-based similarity metric" (README) | **Unsupported** | The only metric computed in any artifact is top-1 accuracy (`preds == labels`). No precision, recall, PR curve, or embedding-similarity metric is implemented. Relabelled as an unrealised proposal. |
| 5 | "Each dataset is cleaned and pruned to remove duplicates, low-quality images, and outliers" (README) | **Unsupported** | No deduplication, filtering or outlier code exists; the notebooks iterate over every image of each split (100,000 / 3,030 / 11,983) and the scripts over every annotation. Removed. |
| 6 | "Data augmentation: RandAug, Mixup, CutMix ... Layer-wise decay" (README, slide 12, W&B) | Partly | Present in both scripts, with defects: validation also used RandAugment and random flips; MixUp used alpha 1.0 (not the stated 0.8) because alphas were never passed; drop path was stated (0.1) but never applied (see `docs/changes.md`). |
| 7 | "Drop Path ... helps avoid overfitting" (slide 12; "drop path rate of 0.1", W&B) | **Incorrect** for these runs | The scripts' `BottleNeck`/`DropPath` substitution loop matches no module in Hiera, so the historical runs trained with stochastic depth 0.0. Verified by replaying the loop against `hiera_tiny_224` (0 `DropPath` modules with p>0). |
| 8 | "300 total epochs, with the first 5 epochs used for warmup" (W&B) | Partly | The scripts are configured for 300 epochs, but the W&B charts stop at step ~89 (iNat2019, epoch ~45) and step ~101 (iNat21-mini, epoch ~52) as of the report date (2024-09-05). The "warmup" was a constant 0.1x learning-rate factor for 5 epochs, not a ramp. Whether any run later reached 300 epochs is not recorded anywhere. |
| 9 | "learning rate of 2e-3 following a cosine decay schedule" (W&B) | Partly | Per-layer rates were set explicitly (head 1.95e-3, decaying by 0.65 per block down to 4.7e-6 for the embeddings); the AdamW `lr=2e-3` argument was overridden by every group. Cosine with `T_max=300`. |
| 10 | "9 Random Augmentations with a magnitude of 0.5" (W&B, iNat21) / "magnitude of 1" (W&B, iNat2019) | Partly | Both scripts use `RandAugment(num_ops=9, magnitude=1)` (torchvision's integer scale, 0-30); the in-code comment says 0.5. Magnitude 1 is a very weak augmentation. |
| 11 | "Mixup and cutmix were used with values of 0.8 and 1.0" (W&B) | **Incorrect** | `v2.MixUp(num_classes=...)` and `v2.CutMix(num_classes=...)` were constructed without `alpha`; torchvision defaults both to 1.0. |

## Datasets

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 12 | "We use three key datasets: Tree of Life 10M+, iNaturalist, Rare Species" (README) | **Incorrect** framing | TreeOfLife-10M was never loaded, read or evaluated on. It is the pre-training set of the external BioCLIP checkpoint. Datasets used directly: iNat 2019 (train + val), iNat 2021 (train_mini + val), Rare Species. |
| 13 | "Tree of Life 10M+: 45,000 species, 10M+ images" (slide 7) | **Incorrect** count | The dataset card states "over 10 million images covering 454 thousand taxa" (8,455,243 images with full 7-rank labels); the species count on the slide is off by a factor of ten. |
| 14 | "iNaturalist 2019: 1010 species, 268,000 images" (slide 7); "300k+ images, 1010 classes" (W&B) | Supported (slide) / Incorrect (W&B) | Upstream: 1,010 species, 72 genera, 268,243 train+val images (265,213 train, 3,030 val). "300k+" is wrong. |
| 15 | "iNaturalist 2021: 10,000 species, 2.7M images, Mini has 500K" (slide 7) | Supported | Upstream: 10,000 species; train 2,686,843; train_mini 500,000 (50 per species); val 100,000 (10 per species); test 500,000. |
| 16 | "500k images, 10 per species/class, resulting in 10,000 different classes" (W&B, iNat21-mini) | **Incorrect** | train_mini has 50 images per species; it is the val split that has 10. |
| 17 | "Rare Species: 400 species, 12,000 images, data from Encyclopedia of Life" (slide 7) | Supported (rounded) | Card: 400 IUCN-listed species, 11,983 images (30 per species nominally; 5 species have fewer than 30 unique images), all from EOL. |
| 18 | "We used the Rare Species Dataset ... as BioCLIP considers these as previously unseen species in its training data" (slide 18) | Supported by upstream statement | The TreeOfLife-10M and Rare Species cards both state the 400 species were removed from TreeOfLife-10M before training. Not independently verifiable from here. |
| 19 | "Tree of Life and iNat have similar species but differ in quality (curated vs. user-generated)" (README) | Partly | TreeOfLife-10M *contains* the whole iNat21 training split (card: "compilation scripts pointed to the training split"), plus BIOSCAN-1M and 6.6M EOL images. They are not two comparable sources; one includes the other. |

## Results

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 20 | Hiera(T) iNat2019 "accuracy peaks at 0.7" (slide 13); "0.69" (W&B text) | Partly | The W&B val_accuracy chart (the only artifact) ends near 0.67 at step ~89; the exact peak is not recoverable. No checkpoint or log file exists in the repository. |
| 21 | Hiera(T) iNat21-mini "accuracy peaked at 0.61" (slide 14, W&B) | Supported (approximate) | Chart peaks at about 0.61 around step 50-60 (epoch ~25-30), then declines to ~0.56 by step ~101. |
| 22 | Validation loss curves (slides 13-14, W&B) | **Invalid** | Hiera's head applies a softmax in eval mode; the scripts then applied `CrossEntropyLoss` to those probabilities. The recorded "val_loss" values (~6.5 for 1,010 classes, ~8.9 for 10,000) are approximately ln(num_classes) and carry no information. Accuracy is unaffected. |
| 23 | "Battle of the CLIPs": CLIP 0.06, OpenCLIP 0.08, BioCLIP 0.70 on iNat 2019 (slide 17) | Partly | BioCLIP 0.7063 is recorded (`openclippy2` notebooks). No artifact exists for the CLIP or OpenCLIP numbers. |
| 24 | BioCLIP(10M+) 0.6398 vs BioCLIP(iNat-21) 0.72 on iNat-21 (slide 20) | Supported, with a methodological issue | Recorded in `BioClip-INAT21-Fixed (2).ipynb` (0.6398, 0.7195). Text embeddings were not L2-normalised, so the scores were not the cosine similarities BioCLIP is evaluated with upstream; the inat-only model was run with the other checkpoint's preprocessing object. |
| 25 | Rare Species text types: 0.294 / 0.313 / 0.327 / 0.36 / 0.371 / 0.373 (slide 19) | Partly | Three of six have notebooks (0.2944, 0.3717, 0.3739). The common-name run is a 398-way task (two collisions merge four species); the others are 400-way, so the bars are not on one task. Scientific, taxonomic and scientific+common have no artifact. |
| 26 | "BioCLIP demonstrates strong performance ... particularly with mixed text styles" (README) | Supported in direction | Among the three recorded runs, taxonomic+common (0.372-0.374) beats common-only (0.294); consistent with the BioCLIP paper. |
| 27 | "BioCLIP was more effective when detailed text annotations were available, while Hiera excelled in purely visual classification" (README) | **Unsupported** as a comparison | Hiera was fine-tuned (supervised) and BioCLIP was zero-shot; they were run on different splits and settings and never compared on one task. |
| 28 | "Refined RAG methodologies" as future work (README); RAG deck | Context only | The RAG deck is Nate Lesperance's 30 April 2024 talk, included for context; no RAG code or result exists in this repository. |

## Documentation accuracy

| # | Claim (source) | Verdict |
|---|---|---|
| 29 | `Notebooks/README.md`: "openclippy2 ... explores the application of OpenCLIP" | **Incorrect**: both notebooks load the BioCLIP checkpoint. |
| 30 | `Notebooks/README.md`: "BioClip-INAT21.ipynb ... original analysis notebook for applying BioCLIP to the iNaturalist [2021] dataset" | **Incorrect**: its recorded result is on iNat 2019 val; the cell that would load 2021 data is syntactically broken. |
| 31 | `iNAT-Model-Code/README.md`: "configured to run for 300 epochs" | Supported for the configuration; see #8 for what was logged. |
