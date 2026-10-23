# Species classification with Hiera and BioCLIP

Research-internship project at the Vector Institute (Toronto), summer 2024, supervised by
Graham Taylor. Two questions: how accurately can a Hiera vision transformer be fine-tuned to
recognise species on the iNaturalist 2019 and 2021 competition datasets, and how well does
BioCLIP classify species zero-shot on iNaturalist and on the Rare Species benchmark, depending
on how the species names are written into the prompt.

The repository was revised in 2026: the training and evaluation code was rewritten, tested and
made runnable on any machine, every 2024 claim was checked against the surviving artifacts, and
the results below carry their caveats inline. **No experiment was re-run; the numbers are the
2024 values.** The original files are kept under [`archive/2024-original/`](archive/2024-original/)
and at the git tag `original-2024`.

![Top-1 accuracy as recorded in 2024](docs/figures/results_top1.png)

## Results

| Experiment | Top-1 | Evidence | Caveat |
|---|---|---|---|
| Hiera-tiny (MAE on ImageNet-1k) fine-tuned on iNat 2019 train, evaluated on val (3,030 images, 1,010 species) | 0.67-0.69 | W&B chart | ~45 of 300 configured epochs logged; validation images were randomly augmented; drop path was 0.0 not 0.1; no checkpoint survives |
| Hiera-tiny fine-tuned on iNat 2021 train_mini, evaluated on val (100,000 images, 10,000 species) | ~0.61 | W&B chart | ~52 epochs logged; same caveats |
| BioCLIP zero-shot, iNat 2019 val, prompt "a photo of a {scientific name}" | 0.706 | notebook output | most of these species are in BioCLIP's pre-training data |
| BioCLIP zero-shot, iNat 2021 val, prompt "{taxonomy} with common name {common}" | 0.640 | notebook output | text embeddings were not normalised; every species is in BioCLIP's pre-training data |
| BioCLIP (iNat21-only checkpoint), same setting | 0.720 | notebook output | as above, plus preprocessing taken from the other checkpoint |
| BioCLIP zero-shot, Rare Species (11,983 images, 400 species), common names | 0.294 | notebook output | 398-way: two shared common names merged four species |
| BioCLIP zero-shot, Rare Species, taxonomy + common name, with / without "a photo of" | 0.372 / 0.374 | notebook output | 400-way; the only unseen-species evaluation in the project |

Three further Rare Species prompt variants (0.31-0.36) and the CLIP / OpenCLIP baselines
(0.06 / 0.08 on iNat 2019) exist on slides only and have no notebook or log. The precision-recall
and embedding-similarity metrics mentioned in the 2024 write-up were never implemented; only
top-1 accuracy was measured. Provenance for every number, including the source cell, is in
[`results/results.md`](results/results.md) (machine-readable: `results/results_inventory.json`);
the claim-by-claim check is in [`docs/claims_audit.md`](docs/claims_audit.md).

Two findings from the audit change how the numbers should be read. BioCLIP's pre-training set,
TreeOfLife-10M, contains the whole iNat21 training split, so the iNaturalist evaluations are
zero-shot only in the no-fine-tuning sense; Rare Species, whose 400 species were removed from
that set, is the one unseen-species test ([`docs/evaluation_validity.md`](docs/evaluation_validity.md)).
And Hiera's "hierarchy" is spatial (four stages of decreasing resolution); the fine-tuning is a
flat softmax over species and uses no taxonomy.

## Demo

Zero-shot classification with the pinned BioCLIP checkpoint against the 400 Rare Species
labels. `--sample` streams a few labelled images from the dataset so the predictions can be
checked against ground truth; `--image` classifies your own photos. The first run downloads the
checkpoint (~600 MB) and, for `--sample`, one dataset shard (~250 MB).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[hub]"
python scripts/demo.py --sample 6
python scripts/demo.py --image my_photo.jpg --prompt-style common --prefix "a photo of "
```

Output of the first command on an 8-core CPU (39 s including model load):

```
checkpoint imageomics/bioclip@ce901ab3c6a9 on cpu; 400 species labels; prompt style 'taxonomic_common', prefix ''

dataset row 0   true: Cyclonaias tuberculata (purple wartyback)
  1. Cyclonaias tuberculata       purple wartyback               cosine 0.262  <- correct
  2. Haliotis roei                Roe's abalone                  cosine 0.191
  3. Wallago attu                 freshwater shark               cosine 0.190
  4. Aonyx capensis               African Clawless Otter         cosine 0.176
  5. Potamilus amphichaenus       Texas heelsplitter             cosine 0.172

dataset row 200   true: Anolis koopmani (Koopmans Anole)
  1. Negaprion acutidens          sicklefin lemon shark          cosine 0.190
  2. Onychoprion aleuticus        Aleutian Tern                  cosine 0.189
  3. Thinornis cucullatus         Hooded Dotterel                cosine 0.185
  4. Rhinoderma darwinii          Darwin's Frog                  cosine 0.183
  5. Negaprion brevirostris       lemon shark                    cosine 0.183

dataset row 400   true: Ceratophora tennentii (Rhinoceros Agama)
  1. Anolis bitectus              Roof Anole                     cosine 0.283
  2. Anolis macrolepis            Big-scaled Anole               cosine 0.244
  3. Anolis baracoae              Baracoa Anole                  cosine 0.241
  4. Anolis luciae                Saint Lucian Anole             cosine 0.230
  5. Anolis barbouri              Hispaniolan Hopping Anole      cosine 0.222

dataset row 600   true: Euryceros prevostii (Helmet Vanga)
  1. Paradisaea rubra             Red Bird-of-Paradise           cosine 0.203
  2. Campylopterus villaviscensio Napo Sabrewing                 cosine 0.193
  3. Mantella aurantiaca          Golden Mantella                cosine 0.191
  4. Ognorhynchus icterotis       Yellow-eared Parrot            cosine 0.188
  5. Sericossypha albocristata    White-capped Tanager           cosine 0.184

dataset row 800   true: Adetomyrma venatrix (dracula ant)
  1. Adetomyrma venatrix          dracula ant                    cosine 0.226  <- correct
  2. Ognorhynchus icterotis       Yellow-eared Parrot            cosine 0.215
  3. Balaeniceps rex              Shoebill                       cosine 0.192
  4. Pontoporia blainvillei       Franciscana                    cosine 0.191
  5. Naultinus stellatus          Starry Tree Gecko              cosine 0.189

dataset row 1000   true: Acropora millepora (staghorn coral)
  1. Stylophora pistillata        Hood Coral                     cosine 0.284
  2. Acropora acuminata           staghorn coral                 cosine 0.270
  3. Acropora millepora           staghorn coral                 cosine 0.268  <- correct
  4. Fungia fungites              Common Mushroom Coral          cosine 0.244
  5. Acropora cervicornis         Staghorn Coral                 cosine 0.233

top-1 correct on these 6 streamed samples: 2/6 (a demo, not an evaluation)
```

Two of six is in line with the recorded 37% on the full set. The last sample is the
shared-common-name case: *Acropora acuminata* and *A. millepora* are both "staghorn coral" and
score within 0.002 of each other.

## What was done

**Hiera fine-tuning.** `hiera_tiny_224` initialised from the MAE ImageNet-1k checkpoint, full
fine-tuning with AdamW (weight decay 0.05), layer-wise learning-rate decay 0.65, label smoothing
0.1, RandAugment, MixUp and CutMix, batch 512, one A40, about 30 minutes per epoch. Runs on
iNat 2019 (265,213 training images) and iNat 2021 train_mini (500,000). The 2024 scripts had
several defects, listed with their effect in [`docs/changes.md`](docs/changes.md); the
corrected recipe is in `configs/hiera_inat2019.yaml` and `configs/hiera_inat2021_mini.yaml`.

**BioCLIP zero-shot evaluation.** One text prompt per species, cosine similarity between image
and text embeddings, argmax. Prompt formats follow the BioCLIP paper: common name, scientific
name, full seven-rank taxonomy, and combinations, with or without a leading "a photo of".
Species identity is kept separate from prompt text, so two species sharing a common name are
scored with an explicit tie rule instead of being merged.

## Repository

| | |
|---|---|
| `src/biodiv/` | Shared code: iNaturalist loading with explicit category maps, train vs. eval transforms, Hiera construction, LR schedule, checkpoint resume, training loop, zero-shot evaluation, pinned checkpoint loading, synthetic fixtures |
| `scripts/` | `demo.py`, `train_hiera.py --config ...`, `eval_bioclip.py --dataset ... --checkpoint ... --prompt-style ...`, `make_results_figure.py` |
| `configs/` | Hiera recipes for both datasets; pinned Hub revisions and dataset paths for evaluation |
| `notebooks/` | `bioclip_inat21_val`, `bioclip_rare_species_prompts`, `inaturalist_tutorial` (output-free; run on fixtures without data) |
| `results/` | Results table, machine-readable inventory, the 2024 W&B report |
| `docs/` | Claims audit, evaluation validity, change log, reproduction guide, environment, licensing, the presentation (`.pptx` and PDF; the slide-17 chart does not render in the PDF export), figures |
| `tests/` | CPU tests on synthetic fixtures and real upstream record samples; CI runs them plus the notebooks |

## Reproducing

```bash
pip install -e ".[dev]" && pytest -q                         # ~30 s, no downloads
python scripts/train_hiera.py --config configs/hiera_inat2019.yaml        # needs the dataset and a GPU
python scripts/eval_bioclip.py --dataset rare-species --checkpoint bioclip --prompt-style taxonomic_common
```

Dataset layout, path configuration (`BIODIV_DATA_ROOT`, `BIODIV_CACHE_DIR`, `BIODIV_OUTPUT_DIR`),
the full command list and the status of each check are in
[`docs/reproduction.md`](docs/reproduction.md). A real run of the corrected code is a new
experiment and should be recorded as such, not as a confirmation of the 2024 values.

## Credits

Supervision: Graham Taylor (Vector Institute, University of Guelph). Thanks to the University
of Guelph Machine Learning Reading Group and the iNaturalist community.

* Hiera: Ryali et al., "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
  ICML 2023 (`facebookresearch/hiera`; code Apache-2.0, weights CC BY-NC 4.0).
* BioCLIP, TreeOfLife-10M and Rare Species: Stevens et al., "BioCLIP: A Vision Foundation Model
  for the Tree of Life", CVPR 2024 (Imageomics Institute; model MIT, datasets CC0).
* iNaturalist 2019 and 2021 competition datasets: Van Horn et al. / Visipedia
  (`visipedia/inat_comp`); non-commercial research use, no image redistribution.
* open_clip (Ilharco et al.), torchvision, Weights & Biases.

Licensed under the MIT License (`LICENSE`); third-party components keep their own licences
(`docs/licensing.md`).
