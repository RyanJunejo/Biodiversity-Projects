# Licensing and attribution

## This repository

Code, notebooks, documentation and the presentation authored by Ryan Junejo are released under
the **MIT License** (see `LICENSE`). The 2024 material under `archive/2024-original/` and the
2026 revision are covered alike.

The presentation (`docs/presentations/`) reproduces a few third-party figures for commentary:
Hiera architecture figures (Ryali et al., 2023), Table 3 of the BioCLIP paper (Stevens et al.,
2024), a GBIF chart, and a SINR figure (Cole et al., 2023). Those remain the property of their
authors and are not covered by the MIT grant.

No dataset images are stored in the repository. Inline images were removed from the archived
notebooks because the iNaturalist terms prohibit redistributing dataset images; the synthetic
test fixtures contain no real photographs.

## Upstream licences (for anything you download or depend on)

| Component | Licence | Notes |
|---|---|---|
| Hiera code (`hiera-transformer`) | Apache-2.0 | code licence relaxed on 2024-03-02 |
| Hiera model weights (`mae_in1k`, etc.) | **CC BY-NC 4.0** | non-commercial; applies to fine-tuned derivatives too |
| BioCLIP checkpoints (`imageomics/bioclip`, `...-inat-only`) | MIT | |
| TreeOfLife-10M (metadata) | CC0-1.0 | images carry their own EOL/iNat licences; iNat21 images are not redistributed by the card |
| Rare Species | CC0-1.0 (dataset card) | images from EOL with per-image licences listed in `licenses.csv` |
| iNaturalist 2019 / 2021 competition data | custom terms | non-commercial research and education only; **no redistribution of images**; see the competition READMEs |
| open_clip | MIT | |
| torchvision, timm, huggingface_hub | BSD-3 / Apache-2.0 | |

## Attribution

Supervision: Graham Taylor (Vector Institute / University of Guelph). Thanks to the University of
Guelph Machine Learning Reading Group, the iNaturalist community and the Visipedia team
(Grant Van Horn et al.) for the competition datasets, the Imageomics Institute for BioCLIP,
TreeOfLife-10M and Rare Species, and Meta AI for Hiera.
