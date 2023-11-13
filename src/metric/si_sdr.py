from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from .base_metric import BaseAudioMetric


class SISDRMetric(BaseAudioMetric):
    def __init__(self, sr=16000, mode="wb"):
        super().__init__(sr, -20)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def _calc_metric(self, estimate_short, target_audio):
        return self.si_sdr(estimate_short, target_audio)
