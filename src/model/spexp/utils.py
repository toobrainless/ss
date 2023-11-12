from torch import nn


class ChannelLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = super().forward(x)
        x = x.transpose(-1, -2)
        return x
