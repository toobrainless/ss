import torch
from torch import nn

from .resnet_block import ResNetBlock
from .utils import ChannelLayerNorm


class SpeakerEncoder(nn.Module):
    def __init__(self, N, speaker_dim, n_classes, num_resnet_blocks=3):
        super().__init__()
        self.channel_layer_norm = ChannelLayerNorm(3 * N)
        self.conv1 = nn.Conv1d(3 * N, speaker_dim, 1)

        self.num_resnet_blocks = num_resnet_blocks
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(speaker_dim) for _ in range(num_resnet_blocks)]
        )
        self.conv2 = nn.Conv1d(speaker_dim, speaker_dim, 1)
        self.linear = nn.Linear(speaker_dim, n_classes)

    def _length_after_resnet(self, length_before):
        return torch.div(
            length_before, (3**self.num_resnet_blocks), rounding_mode="floor"
        )

    def forward(self, x, x_lengths):
        x = self.conv1(self.channel_layer_norm(x))
        x = self.conv2(self.resnet_blocks(x))
        x = x.sum(dim=-1) / self._length_after_resnet(x_lengths)[:, None]
        logits = self.linear(x)
        return x, logits
