from abc import abstractmethod

import pyloudnorm as pyln
import torch

from src.utils import normalize_loud


class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def get_metrics(self):
        return [self.name]

    def __call__(self, **batch):
        raise NotImplementedError()


class BaseAudioMetric(BaseMetric):
    def __init__(self, sr, target_loudness, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter = pyln.Meter(sr)
        self.target_loudness = target_loudness

    def __call__(self, estimate_short, target_audio, *args, **kwargs):
        ans = 0
        for estimate_short_1d, target_audio_1d in zip(estimate_short, target_audio):
            normalize_estimate = normalize_loud(
                estimate_short_1d.detach().cpu().numpy(),
                meter=self.meter,
                target_loudness=self.target_loudness,
            )

            normalize_target = normalize_loud(
                target_audio_1d.detach().cpu().numpy(),
                meter=self.meter,
                target_loudness=self.target_loudness,
            )

            ans += self._calc_metric(
                torch.tensor(normalize_estimate), torch.tensor(normalize_target)
            )

        return ans / len(estimate_short)

    @abstractmethod
    def _calc_metric(self, estimate, target):
        pass
