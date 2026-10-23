# Licensing and attribution (unresolved items flagged, no licence chosen)

The repository had no `LICENSE` file in 2024 and still has none: choosing one is the owner's
decision and may involve the Vector Institute's internship terms. This note records what is
known so that decision can be made.

## Content in this repository

| Content | Author / owner | Licence status |
|---|---|---|
| 2024 scripts, notebooks, README (now under `archive/2024-original/`) | Ryan Junejo, written during a Vector Institute research internship (summer 2024, supervised by Graham Taylor) | **Unresolved**: no licence declared; whether Vector or the University hold rights under the internship agreement is unknown to this cleanup. |
| 2026 code, tests, docs (`src/`, `scripts/`, `tests/`, `docs/`, `notebooks/`, `results/*.md|json`) | written for Ryan Junejo in this cleanup | Same status as above; intended to follow whatever licence is chosen for the repository. |
| `docs/presentations/Advancing_Biodiversity_Conservation_Species_Classification.pptx` (+ PDF export) | Ryan Junejo; Vector Institute slide template | **Unresolved**; contains third-party figures: Hiera paper figures (Ryali et al., 2023), BioCLIP paper Table 3 (Stevens et al., 2024), a GBIF chart, a tweet screenshot (Daniel Bourke), a SINR figure (Cole et al., 2023), stock/organism photos. Redistribution rights for those were not checked. |
| `docs/presentations/RAG_presentation_Nate_Lesperance_2024-04-30.pptx` (+ PDF export) | **Nate Lesperance** (Department of Integrative Biology), not the repository owner | **Unresolved / third-party**: included in 2024 for context; permission to redistribute is not recorded. Consider asking the author or removing it. |
| `results/historical/wandb_report_2024-09-05_iNaturalist_Evaluation_Metrics.pdf` | Ryan Junejo (export of a W&B report) | As the repository. |

No dataset images are stored in the repository. The archived notebooks contain a handful of
embedded output images (one Rare Species image from EOL and three iNat 2019 images rendered by
matplotlib); the iNaturalist terms prohibit redistributing dataset images, so **consider
stripping those outputs from the archive copies** if the repository is to be public (they were
kept here because the brief asked for the originals to be preserved verbatim).

## Upstream licences (for anything you download or depend on)

| Component | Licence | Notes |
|---|---|---|
| Hiera code (`hiera-transformer`) | Apache-2.0 | code licence relaxed on 2024-03-02 |
| Hiera model weights (`mae_in1k`, etc.) | **CC BY-NC 4.0** | non-commercial; applies to the fine-tuned derivatives too |
| BioCLIP checkpoints (`imageomics/bioclip`, `...-inat-only`) | MIT | |
| TreeOfLife-10M (metadata) | CC0-1.0 | images carry their own EOL/iNat licences; iNat21 images are not redistributed by the card |
| Rare Species | CC0-1.0 (dataset card) | images from EOL with per-image licences listed in `licenses.csv` |
| iNaturalist 2019 / 2021 competition data | custom terms | non-commercial research and education only; **no redistribution of images**; see the competition READMEs |
| open_clip | MIT | |
| torchvision, timm, huggingface_hub | BSD-3 / Apache-2.0 | |

## Attribution kept from 2024

Supervision: Graham Taylor (Vector Institute / University of Guelph). Mentorship: Nate
Lesperance. The University of Guelph Machine Learning Reading Group. The iNaturalist community
and the Visipedia team (Grant Van Horn et al.) for the competition datasets; the Imageomics
Institute for BioCLIP, TreeOfLife-10M and Rare Species; Meta AI for Hiera.
