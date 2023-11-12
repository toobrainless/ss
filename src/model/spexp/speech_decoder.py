from torch import nn


class SpeechDecoder(nn.Module):
    def __init__(self, L1, L2, L3, N):
        super().__init__()
        self.short_decoder = nn.ConvTranspose1d(N, 1, L1, L1 // 2)
        self.middle_decoder = nn.Sequential(
            nn.ConvTranspose1d(N, 1, L2, L2 // 2),
        )
        self.long_decoder = nn.Sequential(
            nn.ConvTranspose1d(N, 1, L3, L3 // 2),
        )

    def forward(self, y1, y2, y3):
        y1 = self.short_decoder(y1)
        y2 = self.middle_decoder(y2)
        y3 = self.long_decoder(y3)

        return y1, y2, y3
