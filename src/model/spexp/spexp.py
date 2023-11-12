import torch.nn.functional as F
from torch import nn

from .speaker_encoder import SpeakerEncoder
from .speaker_extractor import SpeakerExtractor
from .speech_decoder import SpeechDecoder
from .speech_encoder import SpeechEncoder


class SpExPlus(nn.Module):
    def __init__(
        self,
        speaker_dim,
        tcn_block_dim,
        tcn_kernel_size,
        tcn_num_blocks,
        tcn_num_stacks,
        L1,
        L2,
        L3,
        N,
        n_classes,
    ):
        super().__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, N)
        self.speaker_encoder = SpeakerEncoder(N, speaker_dim, n_classes)
        self.speaker_extractor = SpeakerExtractor(
            N,
            speaker_dim,
            tcn_block_dim,
            tcn_kernel_size,
            tcn_num_blocks,
            tcn_num_stacks,
        )
        self.speech_decoder = SpeechDecoder(L1, L2, L3, N)

    @staticmethod
    def _shrink_to_fit(x, desired_length):
        if x.shape[-1] >= desired_length:
            return x[..., :desired_length]
        else:
            return F.pad(x, (0, desired_length - x.shape[-1]))

    def forward(self, mix, ref, ref_lengths):
        mix_length = mix.shape[-1]
        mix_encode, y = self.speech_encoder(mix.unsqueeze(1), return_tuple=True)
        ref_encode = self.speech_encoder(ref.unsqueeze(1))
        ref_encode, logits = self.speaker_encoder(
            ref_encode, self.speech_encoder._length_after(ref_lengths)
        )
        y1, y2, y3 = self.speaker_extractor(mix_encode, y, ref_encode)

        short, middle, long = self.speech_decoder(y1, y2, y3)

        short = self._shrink_to_fit(short.squeeze(1), mix_length)
        middle = self._shrink_to_fit(middle.squeeze(1), mix_length)
        long = self._shrink_to_fit(long.squeeze(1), mix_length)

        return {
            "estimate_short": short,
            "estimate_middle": middle,
            "estimate_long": long,
            "logits": logits,
        }
