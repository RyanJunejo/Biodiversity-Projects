"""Accuracy metrics on a ``[n_images, n_classes]`` score matrix."""

from __future__ import annotations

import torch


def top1_correct(scores: torch.Tensor, labels: torch.Tensor, tie_policy: str = "first") -> torch.Tensor:
    """Boolean tensor of per-image correctness.

    tie_policy="first": prediction is the lowest index among maximal scores
    (what ``argmax`` returns), so exact ties count as correct for only one
    class. tie_policy="any": correct when the true class scores equal to the
    maximum, so species sharing identical prompt text are all accepted.
    """
    if tie_policy == "first":
        return scores.argmax(dim=1) == labels
    if tie_policy == "any":
        true_scores = scores.gather(1, labels[:, None]).squeeze(1)
        return true_scores >= scores.max(dim=1).values
    raise ValueError(f"unknown tie_policy {tie_policy!r}")


def topk_accuracy(scores: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    k = min(k, scores.size(1))
    topk = scores.topk(k, dim=1).indices
    return (topk == labels[:, None]).any(dim=1).float().mean().item()


def accuracy(scores: torch.Tensor, labels: torch.Tensor, tie_policy: str = "first") -> float:
    if scores.numel() == 0:
        return float("nan")
    return top1_correct(scores, labels, tie_policy).float().mean().item()
