# Environments

## 2024 (original runs) - partially known

* Vector Institute cluster, SLURM jobs, one NVIDIA A40 per training run (W&B report).
* Conda environment named `hiera`, Python 3.10.14 (from notebook metadata and a traceback path
  `/fs01/home/junejory/miniforge3/envs/hiera/lib/python3.10/`).
* Packages: `torch`, `torchvision` (with `transforms.v2`, so >= 0.15), `open_clip_torch`,
  `transformers`, `datasets`, `wandb`, `tensorboard`, `Pillow`, a third-party `RandAugment`
  package (imported, unused). **Versions were not recorded anywhere.**
* Hiera from `torch.hub.load("facebookresearch/hiera", ...)` at whatever `main` was in summer
  2024; BioCLIP from the Hugging Face Hub at unrecorded revisions.
* Datasets read from `/datasets/inat_comp/{2019,2021}` and scratch copies of annotation files;
  Rare Species downloaded with `datasets` into `/scratch/ssd004/scratch/junejory`.

## 2026 (tests, smoke runs, notebook execution, PDF export) - recorded

* Linux x86_64 sandbox (Amazon Linux 2023), 8 CPU cores, no GPU.
* Python 3.11.17 in a `uv` virtual environment; pins in `requirements.txt`; full
  `pip freeze` in `docs/environment_freeze.txt` (112 packages).
* Key versions: torch 2.14.1+cpu, torchvision 0.29.1+cpu, open_clip_torch 3.3.0,
  hiera-transformer 0.1.4, huggingface_hub 2.3.0, timm 1.0.30, Pillow 12.3.0, numpy 2.4.6,
  pytest 9.1.1, nbclient 0.11.0, nbconvert 7.17.2.
* LibreOffice 26.8.1.1 (headless, RPM build) and poppler 24.08.0 (`pdftoppm`) for the PDF export
  and its visual check.

## Pinned external artifacts

| Artifact | Identifier | Pinned how |
|---|---|---|
| Hiera code | `hiera-transformer==0.1.4` (PyPI; Apache-2.0) | `pyproject.toml` |
| Hiera `mae_in1k` weights | `https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth` (660,554,503 bytes, last-modified 2023-07-21 per HTTP headers; CC BY-NC 4.0) | URL recorded in `src/biodiv/hiera_model.py`; not hashed (not downloaded here) |
| BioCLIP | `imageomics/bioclip` @ `ce901ab3c6a913f9e9ef94ce6d27761069f4f01c` | `configs/bioclip_eval.yaml`, `src/biodiv/encoders.py` |
| BioCLIP iNat21-only | `imageomics/bioclip-vit-b-16-inat-only` @ `9637f081cb39aa7b97e2d651de89bc39b7ae57c8` | same |
| Rare Species | `imageomics/rare-species` @ `7e984e1f68ab70170e7913286ad1ff1a0e5ac370` | `src/biodiv/species_data.py` |
| iNat 2019 / 2021 archives | md5 sums in the upstream READMEs | not downloaded here |

The Hub commits are the ones current on 2026-10-23; they are **not** necessarily the revisions
used in 2024, which were not recorded.
