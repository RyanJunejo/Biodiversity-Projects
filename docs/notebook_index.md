# Notebook index: 2024 names to 2026 replacements

All 2024 notebooks are preserved verbatim (with outputs) in `archive/2024-original/notebooks/`.
The replacements under `notebooks/` are committed **without outputs**; they run from a fresh
kernel in smoke mode on synthetic fixtures (`BIODIV_SMOKE_TEST=1`) and against real data when
the datasets are present (see `docs/reproduction.md`).

| 2024 notebook | What it actually does (from its code and recorded outputs) | Replacement |
|---|---|---|
| `BioClip-INAT21-Fixed (2).ipynb` | BioCLIP and BioCLIP-iNat21-only zero-shot on the iNat21 **val** split, prompt `{taxonomy} with common name {common}`; recorded 0.6398 and 0.7195. Text embeddings not normalised; second model run with first model's preprocessing. | `notebooks/bioclip_inat21_val.ipynb`; `scripts/eval_bioclip.py --dataset inat21-val` |
| `BioClip-INAT21.ipynb` | Despite the name, its only recorded accuracy (0.7063) is BioCLIP on the iNat **2019** val split (1,010 prompts, 95 batches of 32 = 3,030 images). Contains a syntactically broken cell and relies on variables (`annotations`, `clip_labels`) defined in another session. | `scripts/eval_bioclip.py --dataset inat2019-val --prompt-style scientific --prefix "a photo of a "` |
| `openclippy2 (1).ipynb`, `openclippy2 (with results).ipynb` | Despite the name, loads the **BioCLIP** checkpoint (`hf-hub:imageomics/bioclip`), not OpenCLIP. iNat 2019 val, prompt `a photo of a {scientific name}`; recorded 0.7063. The two copies differ only in execution counters and one cleared output. | same as above |
| `rarespecies-common.ipynb` | BioCLIP on Rare Species, prompt `a photo of {common}`; 398 labels because two common-name collisions merge four species; recorded 0.2944 (3528/11983). | `notebooks/bioclip_rare_species_prompts.ipynb` |
| `rarespecies-common-taxi.ipynb` | Prompt `a photo of {taxonomy} with common name {common}`, 400 labels; recorded 0.3717. | same |
| `rarespecies-common-taxi-nointro.ipynb` | Same without the `a photo of` prefix; recorded 0.3739. Identical code otherwise. | same |
| `How to Navigate iNaturalist (1).ipynb` | Tutorial on the Vector cluster copies of iNat 2019 (folder walk, random image, transforms, annotation JSON, category names). | `notebooks/inaturalist_tutorial.ipynb` |
| `Notebooks/README.md` | File descriptions (several inaccurate: e.g. describes the `openclippy2` notebooks as OpenCLIP experiments and claims "various evaluation metrics"). | this file |
