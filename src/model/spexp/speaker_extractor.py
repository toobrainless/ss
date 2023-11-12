from torch import nn

from .tcn import StackedTCN
from .utils import ChannelLayerNorm


class SpeakerExtractor(nn.Module):
    def __init__(
        self,
        N,
        speaker_dim,
        tcn_block_dim,
        tcn_kernel_size,
        tcn_num_blocks,
        tcn_num_stacks,
    ):
        super().__init__()
        self.channel_layer_norm = ChannelLayerNorm(3 * N)
        self.conv1 = nn.Conv1d(3 * N, speaker_dim, 1)
        self.tcn = nn.ModuleList(
            [
                StackedTCN(
                    speaker_dim,
                    speaker_dim,
                    tcn_block_dim,
                    tcn_kernel_size,
                    tcn_num_blocks,
                )
                for _ in range(tcn_num_stacks)
            ]
        )
        self.masks_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(speaker_dim, N, 1),
                    nn.ReLU(),
                )
                for _ in range(3)
            ]
        )

    def forward(self, mix_encode, y, speaker_embedding):
        mix_encode = self.conv1(self.channel_layer_norm(mix_encode))
        for tcn_stack in self.tcn:
            mix_encode = tcn_stack(mix_encode, speaker_embedding)
        masks = [mask_head(mix_encode) for mask_head in self.masks_head]
        return [mask * y for mask, y in zip(masks, y)]
