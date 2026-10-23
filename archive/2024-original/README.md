# Archive of the original 2024 files

Everything in this directory is a copy of the files Ryan Junejo uploaded to this repository
between 4 September and 31 October 2024, including every recorded **text** output of the
notebooks. One edit was made: embedded image outputs (dataset photographs rendered inline,
which the iNaturalist terms do not allow redistributing) were removed from the notebook files;
code, markdown and printed outputs are untouched. The files were otherwise only moved
(`git mv`) so that the cleaned-up layout could take the top level. The complete original tree,
with every output and its original file names and locations, is preserved by the annotated git
tag **`original-2024`** (commit `f609f84`, the last 2024 commit):

```bash
git show original-2024 --stat          # what the repository looked like on 2024-10-31
git checkout original-2024 -- .        # restore the original layout into the working tree
```

Do not treat the code here as correct: `docs/changes.md` lists the defects found in it and
`docs/claims_audit.md` lists which documented claims it does and does not support. The
present-day replacements live under `src/`, `scripts/`, `configs/` and `notebooks/`.

## Contents and where their replacements live

| Original file (2024) | Now at | Replacement (2026) |
|---|---|---|
| `README.md` | `archive/2024-original/README.original.md` | `README.md` |
| `iNAT-Model-Code/inat2019-mfinal-300epochs.py` | `hiera-training/inat2019-mfinal-300epochs.py` | `scripts/train_hiera.py --config configs/hiera_inat2019.yaml` |
| `iNAT-Model-Code/inat2021-mfinal-300.py` | `hiera-training/inat2021-mfinal-300.py` | `scripts/train_hiera.py --config configs/hiera_inat2021_mini.yaml` |
| `iNAT-Model-Code/README.md` | `hiera-training/README.md` | `README.md`, `docs/reproduction.md` |
| `Notebooks/BioClip-INAT21-Fixed (2).ipynb` | `notebooks/BioClip-INAT21-Fixed (2).ipynb` | `notebooks/bioclip_inat21_val.ipynb` |
| `Notebooks/BioClip-INAT21.ipynb` | `notebooks/BioClip-INAT21.ipynb` | (misnamed iNat 2019 run; see `docs/notebook_index.md`) |
| `Notebooks/openclippy2 (1).ipynb`, `openclippy2 (with results).ipynb` | `notebooks/` | `scripts/eval_bioclip.py --dataset inat2019-val` |
| `Notebooks/rarespecies-common.ipynb`, `-taxi.ipynb`, `-taxi-nointro.ipynb` | `notebooks/` | `notebooks/bioclip_rare_species_prompts.ipynb` |
| `Notebooks/How to Navigate iNaturalist (1).ipynb` | `notebooks/How to Navigate iNaturalist (1).ipynb` | `notebooks/inaturalist_tutorial.ipynb` |
| `Notebooks/README.md` | `notebooks/README.md` | `docs/notebook_index.md` |
| `Advancing Biodiversity Conservation ... .pptx` | `docs/presentations/Advancing_Biodiversity_Conservation_Species_Classification.pptx` (+ `.pdf` export) | unchanged |
| `RAGpresentation.pptx` | not carried over (a collaborator's April 2024 talk on retrieval-augmented generation, not part of this project's work); available at the tag | - |
| `iNaturalist Evaluation Metrics _ hiera_inat2021_transfer_learning – Weights & Biases.pdf` | `results/historical/wandb_report_2024-09-05_iNaturalist_Evaluation_Metrics.pdf` | unchanged |

## Files that existed earlier in 2024 and were deleted before this cleanup

The 2024 history (visible with `git log --name-status --diff-filter=D original-2024`) shows
that the following were uploaded on 2024-10-30 and deleted by the author on 2024-10-31; they
are not restored here but remain reachable from the tag's ancestry:

* `inat navigation-21.ipynb` (7.4 MB) and `inat navigation-Stratified-K-Fold.ipynb` (3.8 MB)
* `iNat_2019_github.gif`
* `Trained_Models/ReadME.md` (one line: "This folder contains some of my trained models for
  this Project"; no model files were ever committed)

## Notebook outputs kept here

The archived notebooks keep their recorded text outputs (printed label lists, accuracies,
progress bars, tracebacks). They are the only surviving evidence for the BioCLIP accuracies
listed in `results/results.md`, so they are kept; only inline images were removed (see above).
The cleaned notebooks at the top level are committed without outputs.
