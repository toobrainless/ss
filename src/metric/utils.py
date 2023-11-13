from functools import partial

import torch


def si_sdr(estimated, target, eps=1e-8):
    l2norm = partial(torch.linalg.norm, dim=-1)
    alpha = (target * estimated).sum(dim=-1, keepdim=True) / l2norm(
        target, keepdim=True
    ) ** 2
    return 20 * torch.log10(
        l2norm(alpha * target) / (l2norm(alpha * target - estimated) + eps) + eps
    )
