from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from .base_metric import BaseAudioMetric


class PESQMetric(BaseAudioMetric):
    def __init__(self, sr=16000, mode="wb"):
        super().__init__(sr, -20)
        self.pesq = PerceptualEvaluationSpeechQuality(sr, mode)

    def _calc_metric(self, estimate_short, target_audio):
        try:
            ans = self.pesq(estimate_short, target_audio)
        except Exception:
            ans = -1
        return ans
