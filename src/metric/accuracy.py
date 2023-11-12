from .base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, ref_target, **kwargs):
        return (logits.argmax(dim=-1) == ref_target).float().mean().item()
