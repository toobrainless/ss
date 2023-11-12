from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
        )

        self.head = nn.Sequential(nn.PReLU(), nn.MaxPool1d(3))

    def forward(self, x):
        return self.head(x + self.body(x))
