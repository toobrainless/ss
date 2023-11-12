from torch import nn
from torch.nn import Sequential

from .base_model import BaseModel


class BaselineModel(nn.Module):
    def __init__(self, in_features, out_features, fc_hidden=512, **batch):
        super().__init__()
        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=in_features, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=out_features),
        )

    def forward(self, x, **batch):
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
