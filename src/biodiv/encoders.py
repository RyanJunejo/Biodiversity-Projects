"""Image/text encoders behind a small protocol so evaluation code is testable.

``OpenClipEncoder`` wraps BioCLIP (or any open_clip checkpoint on the Hugging
Face Hub). Passing ``revision`` pins the exact Hub commit: the files are
fetched with ``huggingface_hub.snapshot_download`` and loaded through
open_clip's ``local-dir:`` schema, so the preprocessing and tokenizer always
come from the same checkpoint directory as the weights.
"""

from __future__ import annotations

from typing import Protocol

import torch

KNOWN_CHECKPOINTS = {
    # Hub commit hashes observed on 2026-10-23 (see docs/environment.md).
    "imageomics/bioclip": "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
    "imageomics/bioclip-vit-b-16-inat-only": "9637f081cb39aa7b97e2d651de89bc39b7ae57c8",
}


class Encoder(Protocol):
    preprocess: object  # PIL.Image -> tensor [3, H, W]

    def encode_text(self, texts: list[str]) -> torch.Tensor: ...  # [n, D]
    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...  # [n, D]


def pick_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpenClipEncoder:
    def __init__(self, repo_id: str, revision: str | None = None, cache_dir: str | None = None, device: str | None = None):
        import open_clip

        self.device = pick_device(device)
        self.repo_id, self.revision = repo_id, revision
        if revision:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
            name = f"local-dir:{local_dir}"
        else:
            name = f"hf-hub:{repo_id}"
        model, _, preprocess_val = open_clip.create_model_and_transforms(name, cache_dir=cache_dir)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess_val  # evaluation preprocessing of *this* checkpoint
        self.tokenizer = open_clip.get_tokenizer(name)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            return self.model.encode_text(tokens).float()

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            return self.model.encode_image(images.to(self.device)).float()

    def describe(self) -> dict:
        return {"repo_id": self.repo_id, "revision": self.revision, "device": str(self.device)}


class LookupEncoder:
    """Deterministic stand-in for tests and smoke runs (no weights, no network).

    Text embeddings come from ``text_table`` (prompt -> vector); unseen prompts
    get a seeded random vector. Image embeddings are produced by
    ``image_fn(images)``; the default maps each image's mean RGB colour to a
    vector through ``colour_table``.
    """

    def __init__(self, dim: int = 8, text_table: dict[str, torch.Tensor] | None = None, image_fn=None, seed: int = 0):
        from torchvision import transforms

        self.dim = dim
        self.text_table = dict(text_table or {})
        self.image_fn = image_fn
        self.generator = torch.Generator().manual_seed(seed)
        self.preprocess = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
        self.device = torch.device("cpu")

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        out = []
        for t in texts:
            if t not in self.text_table:
                self.text_table[t] = torch.randn(self.dim, generator=self.generator)
            out.append(self.text_table[t])
        return torch.stack(out)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_fn is not None:
            return self.image_fn(images)
        mean_rgb = images.mean(dim=(2, 3))  # [n, 3]
        return torch.nn.functional.pad(mean_rgb, (0, self.dim - 3))

    def describe(self) -> dict:
        return {"repo_id": "lookup-encoder (fixture)", "revision": None, "device": "cpu"}
