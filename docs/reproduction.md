# Reproduction status and how to run things

## Status at a glance (October 2026)

| What | Status |
|---|---|
| Unit / failure-mode tests (`pytest`) | **Pass** (33 tests, CPU, ~30 s; includes adapters checked against real upstream record samples) |
| Hiera training loop smoke test (synthetic 3-class fixture, random init, 2 epochs + resume, CPU) | **Pass** (~10 s) |
| BioCLIP evaluation smoke test (synthetic 4-species fixture, weight-free encoder) | **Pass** |
| Cleaned notebooks executed from a fresh kernel in smoke mode | **Pass** (all three, `jupyter nbconvert --execute`) |
| Hiera fine-tuning on iNat 2019 / iNat 2021-mini re-run | **Not done** (needs the datasets, ~74 GB / ~42 GB, and a GPU; the 2024 runs took ~30 min/epoch on an A40) |
| BioCLIP evaluations re-run on real data | **Not done** (needs the datasets and ~600 MB per checkpoint download; a CPU run over 100,000 iNat21 val images is slow) |

A smoke test proves the code path executes; it is **not** a reproduction and its numbers mean
nothing. A real run of the corrected code is a new experiment (see `docs/changes.md` for what
differs from 2024) and should be recorded as "newly reproduced" in `results/results_inventory.json`
with its own provenance, never as a confirmation of the 2024 values.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"            # package + test/notebook tooling
pip install -e ".[hub,wandb]"      # optional: Rare Species from the Hub, W&B logging
pytest -q
```

Exact versions used for the checks above are in `requirements.txt` and `docs/environment.md`.

## Paths

Nothing is hard-coded. Set:

```bash
export BIODIV_DATA_ROOT=/path/to/data      # contains inat_comp/2019/... and inat_comp/2021/...
export BIODIV_CACHE_DIR=/path/to/cache     # Hugging Face / torch downloads
export BIODIV_OUTPUT_DIR=/path/to/outputs  # checkpoints, logs, result JSON
```

Dataset paths inside `configs/*.yaml` are relative to `BIODIV_DATA_ROOT` unless absolute. On the
Vector cluster the 2024 runs used `/datasets/inat_comp/<year>/` (set `BIODIV_DATA_ROOT=/datasets`).

## Data

* iNaturalist 2019 and 2021: download from the links in the
  [visipedia/inat_comp](https://github.com/visipedia/inat_comp) READMEs (md5 sums given there).
  For 2019 also fetch `categories.json` (un-obfuscated names). Expected layout:
  `inat_comp/2019/{train_val2019/, train2019.json, val2019.json, categories.json}` and
  `inat_comp/2021/{train_mini/, val/, train_mini.json, val.json}`.
* Rare Species: `pip install -e ".[hub]"` and the code downloads `imageomics/rare-species` at
  the pinned revision (about 2 GB), or point `BIODIV_RARE_SPECIES_JSONL` / the config at a JSONL
  export made with `datasets.Dataset.to_json` (the 2024 format).

## Smoke tests (no data, no weights, CPU)

```bash
python -m biodiv.fixtures tests/fixtures/data                          # writes the synthetic iNat fixture (git-ignored)
python scripts/train_hiera.py --config tests/fixtures/hiera_smoke.yaml --data-root tests/fixtures/data
python scripts/eval_bioclip.py --dataset fixture --checkpoint lookup --prompt-style common --prefix "a photo of "
cd notebooks && BIODIV_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute --output-dir /tmp/nb *.ipynb
```

## Real runs

```bash
# Hiera fine-tuning (GPU). Resumes automatically from outputs/<run_name>/checkpoints/last.pt.
python scripts/train_hiera.py --config configs/hiera_inat2019.yaml [--wandb]
python scripts/train_hiera.py --config configs/hiera_inat2021_mini.yaml [--wandb]

# BioCLIP zero-shot (GPU recommended). Checkpoints are pinned in configs/bioclip_eval.yaml.
python scripts/eval_bioclip.py --dataset inat21-val    --checkpoint bioclip             --prompt-style taxonomic_common --prefix ""            --output results/runs/bioclip_inat21_val.json
python scripts/eval_bioclip.py --dataset inat21-val    --checkpoint bioclip-inat21-only --prompt-style taxonomic_common --prefix ""            --output results/runs/bioclip_inat_only_inat21_val.json
python scripts/eval_bioclip.py --dataset inat2019-val  --checkpoint bioclip             --prompt-style scientific       --prefix "a photo of a" --output results/runs/bioclip_inat2019_val.json
python scripts/eval_bioclip.py --dataset rare-species  --checkpoint bioclip             --prompt-style common           --prefix "a photo of "  --output results/runs/bioclip_rare_common.json
python scripts/eval_bioclip.py --dataset rare-species  --checkpoint bioclip             --prompt-style taxonomic_common --prefix "a photo of "  --output results/runs/bioclip_rare_taxcommon.json
python scripts/eval_bioclip.py --dataset rare-species  --checkpoint bioclip             --prompt-style taxonomic_common --prefix ""             --output results/runs/bioclip_rare_taxcommon_noprefix.json
```

(`--prefix "a photo of a"` needs a trailing space: pass `"a photo of a "`.) Use `--limit N` for
a quick partial check. Each run writes a JSON with the counts, both tie policies, skipped
images, the checkpoint revision and the prompt example, which is what `results/` needs for a
"newly reproduced" entry.

To match the historical training behaviour as closely as the corrected code allows, set
`drop_path_rate: 0.0` and `mixup_alpha: 1.0` in the config; the validation transform,
learning-rate schedule and validation loss will still differ (see `docs/changes.md`).

## Record of the checks performed on 2026-10-23

* `pytest -q`: 33 passed (Python 3.11.17, torch 2.14.1+cpu, 8-core sandbox); 25 in the first pass, 8 added after the review described in `docs/changes.md`.
* `scripts/train_hiera.py` smoke config: 2 epochs on 6 synthetic images, checkpoints
  `epoch_0000.pt`, `epoch_0001.pt`, `last.pt`, `history.jsonl`, `final_model.pt` written; the
  resume path was exercised by the test suite (third epoch resumed from `epoch_0001.pt`).
* `scripts/eval_bioclip.py --dataset fixture --checkpoint lookup`: 4 species, 3 unique common
  prompts, `top1_accuracy_first` 0.75 and `top1_accuracy_any` 1.0 as designed (one coral species
  loses the tie under `first`).
* Notebooks: `bioclip_inat21_val`, `bioclip_rare_species_prompts`, `inaturalist_tutorial`
  executed without error from a fresh kernel with `BIODIV_SMOKE_TEST=1`.
* Presentation PDF export: both decks converted with LibreOffice 26.8.1 (headless) and checked
  page by page against the slide text and media (26 and 28 pages). One defect: on slide 17
  ("Battle of the CLIPs") the bar chart's bars do not render and its axis is mis-scaled in the
  PDF; the values (CLIP 0.06, BioCLIP 0.70, OpenCLIP 0.08) are in `results/results.md`. Open the
  `.pptx` for that chart.
