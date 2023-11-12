import torch
from torch import nn


class TCNBase(nn.Module):
    def __init__(
        self, in_channels, speaker_channels, out_channels, dilation, kernel_size
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels + speaker_channels, out_channels, 1),
            nn.PReLU(),
            nn.GroupNorm(1, out_channels),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.PReLU(),
            nn.GroupNorm(1, out_channels),
            nn.Conv1d(out_channels, in_channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class TCNBlock(TCNBase):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super().__init__(in_channels, 0, out_channels, dilation, kernel_size)


class FirstTCNBlock(TCNBase):
    def __init__(self, in_channels, speaker_dim, out_channels, dilation, kernel_size):
        super().__init__(in_channels, speaker_dim, out_channels, dilation, kernel_size)

    def forward(self, x, speaker_embedding):
        time_length = x.shape[-1]
        speaker_embedding = torch.unsqueeze(speaker_embedding, -1)
        speaker_embedding = speaker_embedding.repeat(1, 1, time_length)
        return x + self.net(torch.cat([x, speaker_embedding], dim=1))


class StackedTCN(nn.Module):
    def __init__(self, in_channels, speaker_dim, out_channels, kernel_size, num_blocks):
        super().__init__()
        self.first_block = FirstTCNBlock(
            in_channels, speaker_dim, out_channels, 1, kernel_size
        )

        self.rest_blocks = nn.Sequential(
            *[
                TCNBlock(out_channels, out_channels, 2**i, kernel_size)
                for i in range(1, num_blocks)
            ]
        )

    def forward(self, x, speaker_embedding):
        x = self.first_block(x, speaker_embedding)
        return self.rest_blocks(x)
