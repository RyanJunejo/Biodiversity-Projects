"""Validation preprocessing must be deterministic; training preprocessing may not be."""

import torch
from torchvision import transforms

from biodiv.transforms import build_eval_transform, build_train_transform
from tests.fixtures.make_fixture import make_image


def test_eval_transform_is_deterministic_and_never_flips():
    tf = build_eval_transform(32)
    img = make_image((200, 30, 30), 32, marker=True)  # white marker top-left
    a, b = tf(img), tf(img)
    assert torch.equal(a, b)
    # marker stays top-left: brightest pixel is at column 0, not the last column
    assert a[:, 0, 0].sum() > a[:, 0, -1].sum()
    kinds = {type(t) for t in tf.transforms}
    assert not kinds & {transforms.RandomHorizontalFlip, transforms.RandAugment}


def test_train_transform_contains_the_random_ops():
    tf = build_train_transform(32, True, 9, 1)
    kinds = [type(t) for t in tf.transforms]
    assert transforms.RandomHorizontalFlip in kinds and transforms.RandAugment in kinds
    img = make_image((200, 30, 30), 32)
    outs = [tf(img) for _ in range(20)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])  # augmentation does something
