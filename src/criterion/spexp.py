from torch import nn

from .utils import si_sdr


class SpExPlusCriterion(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        estimate_short,
        estimate_middle,
        estimate_long,
        target_audio,
        target_class,
        logits,
        return_ce=False,
    ):
        estimate_short = estimate_short - estimate_short.mean(dim=-1, keepdim=True)
        estimate_middle = estimate_middle - estimate_middle.mean(dim=-1, keepdim=True)
        estimate_long = estimate_long - estimate_long.mean(dim=-1, keepdim=True)

        si_sdr_short = si_sdr(estimate_short, target_audio)
        si_sdr_middle = si_sdr(estimate_middle, target_audio)
        si_sdr_long = si_sdr(estimate_long, target_audio)

        si_sdr_loss = -(
            (1 - self.alpha - self.beta) * si_sdr_short.sum()
            + self.alpha * si_sdr_middle.sum()
            + self.beta * si_sdr_long.sum()
        )

        if not return_ce:
            return si_sdr_loss

        ce_loss = self.ce(logits, target_class)

        return si_sdr_loss + self.gamma * ce_loss
