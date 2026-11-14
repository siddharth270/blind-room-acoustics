"""Models.

* `CRNN` - teacher: 2-D conv front-end + BiGRU + attention pooling (~0.9 M params).
* `TinyTCN` - deployable student: strided 2-D stem + dilated depthwise-separable
  1-D convolutions + attention pooling. No recurrence, so it quantises cleanly to
  int8 and maps onto CMSIS-NN / TFLite-Micro style kernels.

Both take log-mel input (B, n_mels, T) and regress N_TARGETS values.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AttentionPool(nn.Module):
    """Softmax-weighted average over time; lets the model focus on decay regions."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Conv1d(dim, 1, 1)

    def forward(self, x):  # (B, C, T) -> (B, C)
        w = torch.softmax(self.score(x), dim=-1)
        return (x * w).sum(-1)


class ConvBlock2d(nn.Sequential):
    def __init__(self, cin, cout, pool):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )


class CRNN(nn.Module):
    def __init__(self, n_mels=64, n_out=9, channels=(32, 64, 128, 128), gru=128, dropout=0.2):
        super().__init__()
        pools = [(2, 2), (2, 2), (2, 1), (2, 1)]
        layers, cin = [], 1
        for c, p in zip(channels, pools):
            layers.append(ConvBlock2d(cin, c, p))
            cin = c
        self.conv = nn.Sequential(*layers)
        f_out = n_mels // 16
        self.proj = nn.Linear(cin * f_out, gru)
        self.gru = nn.GRU(gru, gru, batch_first=True, bidirectional=True)
        self.pool = AttentionPool(2 * gru)
        self.head = nn.Sequential(nn.Linear(2 * gru, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_out))

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))                     # (B, C, F, T)
        z = z.flatten(1, 2).transpose(1, 2)               # (B, T, C*F)
        z, _ = self.gru(F.relu(self.proj(z)))
        return self.head(self.pool(z.transpose(1, 2)))


class DSResBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)))
        y = self.drop(self.bn2(self.pw(y)))
        return F.relu(x + y)


class TinyTCN(nn.Module):
    def __init__(self, n_mels=64, n_out=9, stem=(8, 16), width=64, dilations=(1, 2, 4, 8),
                 kernel=5, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(stem[0]), nn.ReLU(inplace=True),
            nn.Conv2d(stem[0], stem[1], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(stem[1]), nn.ReLU(inplace=True),
        )
        f_out = (n_mels + 3) // 4
        self.proj = nn.Sequential(nn.Conv1d(stem[1] * f_out, width, 1, bias=False), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[DSResBlock(width, kernel, d, dropout) for d in dilations])
        self.pool = AttentionPool(width)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        z = self.stem(x.unsqueeze(1)).flatten(1, 2)       # (B, C*F, T/4)
        z = self.blocks(self.proj(z))
        return self.head(self.pool(z))


REGISTRY = {"crnn": CRNN, "tcn": TinyTCN}


def build_model(name: str, cfg: dict, n_out: int) -> nn.Module:
    return REGISTRY[name](n_mels=cfg["features"]["n_mels"], n_out=n_out, **cfg["models"][name])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates for Conv/Linear/GRU layers (forward hooks)."""
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        k = m.weight[0].numel()  # (cin/groups) * prod(kernel)
        macs += out.numel() * k

    def lin_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() * m.in_features

    def gru_hook(m, inp, out):
        nonlocal macs
        T = inp[0].shape[1]
        d = 2 if m.bidirectional else 1
        macs += inp[0].shape[0] * T * d * 3 * m.hidden_size * (m.input_size + m.hidden_size)

    hooks = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.GRU):
            hooks.append(mod.register_forward_hook(gru_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return int(macs)


def masked_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 that ignores NaN targets (e.g. unknown volume for measured RIRs)."""
    mask = ~torch.isnan(target)
    t = torch.nan_to_num(target)
    w = mask.float() * weights
    loss = F.smooth_l1_loss(pred, t, reduction="none", beta=0.5) * w
    return loss.sum() / w.sum().clamp(min=1.0)
