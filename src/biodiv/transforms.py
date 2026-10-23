"""Image transforms.

``build_train_transform`` reproduces the 2024 augmentation pipeline (direct
resize to a square, horizontal flip, RandAugment, ImageNet normalisation).
``build_eval_transform`` contains only deterministic operations; the 2024
scripts evaluated with the training pipeline, including random flips and
RandAugment, which this module fixes.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    image_size: int = 224,
    horizontal_flip: bool = True,
    randaugment_num_ops: int = 9,
    randaugment_magnitude: int = 1,
) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if randaugment_num_ops > 0:
        ops.append(transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic: resize, tensor, normalise. No flips, no RandAugment."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
