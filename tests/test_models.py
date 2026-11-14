import torch

from roomest.config import load_config
from roomest.models import build_model, count_params, masked_loss


def test_models_forward_and_size():
    cfg = load_config("configs/default.yaml")
    x = torch.randn(2, cfg["features"]["n_mels"], 397)
    for name in ("crnn", "tcn"):
        m = build_model(name, cfg, 9).eval()
        assert m(x).shape == (2, 9)
    assert count_params(build_model("tcn", cfg, 9)) < 150_000


def test_masked_loss_ignores_nan():
    p = torch.zeros(2, 3)
    t = torch.tensor([[0.0, float("nan"), 0.0], [0.0, 0.0, float("nan")]])
    assert masked_loss(p, t, torch.ones(3)).item() == 0.0
