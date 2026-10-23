"""Hiera construction.

Uses the ``hiera-transformer`` package (pinned in requirements) instead of an
unpinned ``torch.hub.load``. Two behaviours differ from the 2024 scripts:

* Stochastic depth is requested through Hiera's own ``drop_path_rate``
  argument. The 2024 scripts defined a ``BottleNeck`` class and tried to swap
  it into the model, but Hiera contains no such module, so the loop matched
  nothing and the historical runs trained with drop path effectively 0.0.
* Hiera's classification head applies a softmax whenever the model is in eval
  mode. The 2024 validation loop fed those probabilities into
  ``CrossEntropyLoss`` (a second softmax), which is why the recorded validation
  losses sit at roughly ln(num_classes). ``logits_in_eval=True`` replaces the
  head activation with the identity so eval outputs are logits. Accuracy is
  unaffected either way because softmax preserves the argmax.
"""

from __future__ import annotations

import torch.nn as nn

CHECKPOINT_URLS = {
    # From hiera/hiera.py in hiera-transformer 0.1.4 (facebookresearch/hiera).
    ("hiera_tiny_224", "mae_in1k"): "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
    ("hiera_tiny_224", "mae_in1k_ft_in1k"): "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
}


def _identity(x):
    return x


def build_hiera(
    num_classes: int,
    name: str = "hiera_tiny_224",
    checkpoint: str = "mae_in1k",
    drop_path_rate: float = 0.1,
    pretrained: bool = True,
    logits_in_eval: bool = True,
) -> nn.Module:
    import hiera  # imported lazily so the rest of the package works without it

    factory = getattr(hiera, name)
    model = factory(
        pretrained=pretrained,
        checkpoint=checkpoint,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    if logits_in_eval:
        model.head.act_func = _identity
    return model


def drop_path_rates(model: nn.Module) -> list[float]:
    """Per-block stochastic-depth probabilities actually present in ``model``."""
    rates = []
    for module in model.modules():
        if type(module).__name__ == "DropPath":
            rates.append(float(getattr(module, "drop_prob", 0.0)))
    return rates
