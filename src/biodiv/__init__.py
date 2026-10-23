"""Shared code for the 2024 Vector Institute biodiversity classification project.

Modules
-------
paths        configurable dataset / cache / output locations
inat         iNaturalist competition JSON parsing, explicit category mapping, dataset
transforms   training vs. deterministic evaluation image transforms
hiera_model  Hiera construction with supported drop-path and logits output
optim        layer-wise learning-rate groups and warmup + cosine schedule
checkpoint   checkpoint save / resume that works outside SLURM
train        config-driven Hiera fine-tuning loop (optional W&B logging)
prompts      text prompt formats used for CLIP-style zero-shot classification
encoders     open_clip (BioCLIP) wrapper with explicit checkpoint revisions
clip_eval    zero-shot evaluation with explicit species identity and tie handling
species_data adapters that turn iNat21 / Rare Species metadata into species tables
"""

__version__ = "0.1.0"
