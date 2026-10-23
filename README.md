# Species classification with Hiera and BioCLIP (Vector Institute, summer 2024)

Code, notebooks, slides and recorded results from Ryan Junejo's research internship at the
Vector Institute (Toronto) in summer 2024, supervised by Graham Taylor and mentored by Nate
Lesperance. The work fine-tuned a Hiera vision transformer on the iNaturalist 2019 and 2021
competition datasets and measured BioCLIP's zero-shot species classification on iNaturalist and
on the Rare Species benchmark, with attention to how the text prompt is written.

In October 2026 the repository was reorganised and audited. The original 2024 files are kept
unchanged under [`archive/2024-original/`](archive/2024-original/) and at the git tag
`original-2024`; everything else is new. **No experiment was re-run**: all numbers quoted here
are the 2024 values, with their provenance and known defects stated.

## Research questions

1. How accurately can Hiera-tiny (self-supervised MAE pre-training on ImageNet-1k) be fine-tuned
   to classify species on iNaturalist 2019 (1,010 species) and iNaturalist 2021 "mini"
   (10,000 species)?
2. How well does BioCLIP classify species zero-shot (no fine-tuning) on the iNaturalist 2019 and
   2021 validation splits and on the Rare Species dataset, and which text format for the class
   names works best (common name, scientific name, full taxonomy, combinations, with or without a
   leading "a photo of")?
3. (Context, not pursued in code here) How would these compare with the image-captioning plus
   retrieval-augmented-generation pipeline Nate Lesperance presented in April 2024?

Note on Hiera: its "hierarchy" is spatial (the network pools tokens through four stages of
decreasing resolution). It has nothing to do with biological taxonomy, and the fine-tuning here
is a flat softmax over species. The 2024 README suggested otherwise; that was wrong.

## What was completed

| Experiment | Recorded result (2024) | Evidence | Caveats |
|---|---|---|---|
| Hiera-tiny fine-tuned on iNat 2019 train, evaluated on val (3,030 images) | top-1 about 0.67-0.69 | W&B report chart (`results/historical/`), slide 13 | ~45 of 300 configured epochs logged; validation images were randomly augmented; drop path was 0.0 not 0.1; no checkpoint survives |
| Hiera-tiny fine-tuned on iNat 2021 train_mini, evaluated on val (100,000 images) | top-1 peak about 0.61 | W&B report chart, slide 14 | ~52 epochs logged; same caveats |
| BioCLIP zero-shot, iNat 2019 val, prompt "a photo of a {scientific name}" | 0.7063 | archived notebook output | all 1,010 species are probably in BioCLIP's pre-training data |
| BioCLIP zero-shot, iNat 2021 val, prompt "{taxonomy} with common name {common}" | 0.6398 | archived notebook output | text embeddings not normalised; all 10,000 species are in BioCLIP's pre-training data |
| BioCLIP iNat21-only checkpoint, same setting | 0.7195 | archived notebook output | as above, plus wrong preprocessing object |
| BioCLIP zero-shot, Rare Species, common names | 0.2944 | archived notebook output | 398-way task (two shared common names) |
| BioCLIP zero-shot, Rare Species, taxonomy + common name, with / without "a photo of" | 0.3717 / 0.3739 | archived notebook outputs | 400-way; only genuinely unseen-species evaluation in the project |

Full table with sources and issues: [`results/results.md`](results/results.md); machine-readable:
[`results/results_inventory.json`](results/results_inventory.json). Three further prompt variants
and two CLIP/OpenCLIP baselines appear on slides only and have no artifact. The "precision-recall"
and "embedding similarity" metrics mentioned in the 2024 README were never implemented; only
top-1 accuracy was measured. No deduplication or image filtering was performed.

## Where things are

| | |
|---|---|
| `src/biodiv/` | Shared Python code: iNaturalist loading with explicit category maps, train vs. eval transforms, Hiera construction, layer-wise LR + warmup/cosine schedule, checkpoint resume, training loop, BioCLIP zero-shot evaluation with explicit species identity and tie handling, pinned checkpoint loading |
| `scripts/` | `train_hiera.py --config ...` and `eval_bioclip.py --dataset ... --checkpoint ... --prompt-style ...` |
| `configs/` | `hiera_inat2019.yaml`, `hiera_inat2021_mini.yaml`, `bioclip_eval.yaml` (pinned Hub revisions, dataset paths) |
| `notebooks/` | Reader-facing, output-free: `bioclip_inat21_val.ipynb`, `bioclip_rare_species_prompts.ipynb`, `inaturalist_tutorial.ipynb`. Old-name to new-name map: `docs/notebook_index.md` |
| `results/` | `results.md`, `results_inventory.json`, `historical/` (W&B report PDF) |
| `docs/` | `claims_audit.md` (every 2024 claim checked), `evaluation_validity.md` (what each checkpoint had seen), `changes.md` (code fixes and which change behaviour), `reproduction.md`, `environment.md`, `licensing.md`, `presentations/` (both decks as `.pptx` and PDF) |
| `archive/2024-original/` | Verbatim 2024 scripts, notebooks (with outputs) and README; see its README |
| `tests/` | 25 CPU tests on synthetic fixtures; `.github/workflows/ci.yml` runs them plus the notebooks in smoke mode |

The presentation is `docs/presentations/Advancing_Biodiversity_Conservation_Species_Classification.pdf`
(26 slides; the slide-17 bar chart does not render in the PDF export, open the `.pptx`).

## Known limitations and reproduction status

* The 2024 training scripts evaluated on augmented validation images, applied no stochastic
  depth despite reporting 0.1, used MixUp alpha 1.0 despite reporting 0.8, and computed an
  invalid validation loss (Hiera's eval-mode softmax was fed to cross-entropy). Accuracies are
  unaffected by the last point. Details and the effective learning rates: `docs/changes.md`.
* The iNat21 BioCLIP numbers were computed without normalising the text embeddings.
* The three Rare Species prompt results are not on one task (398 vs 400 labels).
* The W&B charts show the Hiera runs at roughly 45 and 52 epochs on 2024-09-05; whether the
  300-epoch runs completed is not recorded. The trained weights were never committed.
* BioCLIP's pre-training set (TreeOfLife-10M) contains the whole iNat21 training split, so the
  iNaturalist evaluations are zero-shot only in the no-fine-tuning sense; Rare Species is the only
  unseen-species evaluation. Overlap between iNat 2019 and iNat21 images is unknown.
  See `docs/evaluation_validity.md`.
* Package versions and Hub revisions from 2024 were not recorded. The corrected code pins
  current ones (`docs/environment.md`).
* Reproduction status: tests and smoke runs pass on CPU; no real-data run has been made with
  the corrected code. `docs/reproduction.md` gives the commands and data layout.

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest -q                                                   # 25 tests, synthetic fixtures, ~15 s
python tests/fixtures/make_fixture.py && python scripts/train_hiera.py --config tests/fixtures/hiera_smoke.yaml --data-root tests/fixtures/data
python scripts/eval_bioclip.py --dataset fixture --checkpoint lookup --prompt-style common --prefix "a photo of "
```

Real runs need `BIODIV_DATA_ROOT` pointing at the iNaturalist competition data (and a GPU for
training); see `docs/reproduction.md`. W&B logging is off by default (`--wandb` to enable).

## Credits and citations

Supervision: Graham Taylor (Vector Institute, University of Guelph). Mentorship: Nate
Lesperance. Thanks to the University of Guelph Machine Learning Reading Group and the
iNaturalist community.

* Hiera: Ryali et al., "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles",
  ICML 2023. Code Apache-2.0, weights CC BY-NC 4.0 (`facebookresearch/hiera`).
* BioCLIP, TreeOfLife-10M and Rare Species: Stevens et al., "BioCLIP: A Vision Foundation Model
  for the Tree of Life", CVPR 2024 (Imageomics Institute; model MIT, datasets CC0).
* iNaturalist 2019 and 2021 competition datasets: Van Horn et al. / Visipedia
  (`visipedia/inat_comp`); non-commercial research use, no image redistribution.
* open_clip (Ilharco et al.), torchvision RandAugment / MixUp / CutMix, Weights & Biases.

Licensing of this repository's own content is unresolved; see `docs/licensing.md`.
