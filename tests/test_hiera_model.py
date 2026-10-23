"""Hiera is built through supported arguments and returns logits in eval mode."""

import pytest
import torch

hiera = pytest.importorskip("hiera")

from biodiv.hiera_model import build_hiera, drop_path_rates  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_hiera(num_classes=5, pretrained=False, drop_path_rate=0.1)


def test_eval_outputs_are_logits_not_probabilities(model):
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5)
    assert not torch.allclose(out.sum(dim=1), torch.ones(2))  # softmax would sum to 1
    stock = hiera.hiera_tiny_224(pretrained=False, num_classes=5).eval()
    with torch.no_grad():
        assert torch.allclose(stock(x).sum(dim=1), torch.ones(2), atol=1e-5)  # the behaviour we bypass


def test_drop_path_is_actually_configured(model):
    rates = drop_path_rates(model)
    assert rates and max(rates) == pytest.approx(0.1) and len(set(rates)) > 1
    assert drop_path_rates(hiera.hiera_tiny_224(pretrained=False, num_classes=5)) == []
