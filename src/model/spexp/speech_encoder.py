import torch
from torch import nn


class SpeechEncoder(nn.Module):
    def __init__(self, L1, L2, L3, N):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.N = N

        self.short_encoder = nn.Conv1d(1, N, L1, L1 // 2)
        self.middle_encoder = nn.Sequential(
            nn.ConstantPad1d((0, (L2 - L1)), 0), nn.Conv1d(1, N, L2, L1 // 2)
        )
        self.long_encoder = nn.Sequential(
            nn.ConstantPad1d((0, (L3 - L1)), 0), nn.Conv1d(1, N, L3, L1 // 2)
        )

    def forward(self, x, return_tuple=False):
        x1 = self.short_encoder(x)
        x2 = self.middle_encoder(x)
        x3 = self.long_encoder(x)

        assert x1.shape == x2.shape == x3.shape
        if return_tuple:
            return torch.cat([x1, x2, x3], dim=1), (x1, x2, x3)
        return torch.cat([x1, x2, x3], dim=1)

    def _length_after(self, length_before):
        return (
            torch.div(length_before - self.L1, self.L1 // 2, rounding_mode="floor") + 1
        )
