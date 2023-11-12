from dataclasses import dataclass

import pytest
import torch

from src.model import SpeakerEncoder, SpeechEncoder


@dataclass
class LengthSpeechEncoderCases:
    N: int
    L1: int
    L2: int
    L3: int
    length: int


LENGTH_SPEECH_ENCODER_CASES = [
    LengthSpeechEncoderCases(256, 20, 80, 160, 16000),
    LengthSpeechEncoderCases(256, 40, 160, 320, 32000),
    LengthSpeechEncoderCases(256, 10, 40, 80, 5000),
    LengthSpeechEncoderCases(256, 20, 30, 40, 320),
    LengthSpeechEncoderCases(256, 20, 200, 50, 1000),
]


@pytest.mark.parametrize("case", LENGTH_SPEECH_ENCODER_CASES)
def test_length_after_speech_encoder(case: LengthSpeechEncoderCases):
    speech_encoder = SpeechEncoder(case.N, case.L1, case.L2, case.L3)
    batch = torch.randn((5, 1, case.length))
    output = speech_encoder(batch)
    assert output.shape[-1] == speech_encoder._length_after(case.length)


@dataclass
class LengthResNetCases:
    num_resnet_blocks: int
    length: int


LENGTH_RESNET_CASES = [
    LengthResNetCases(1, 160),
    LengthResNetCases(2, 320),
    LengthResNetCases(3, 500),
    LengthResNetCases(4, 480),
    LengthResNetCases(5, 250),
]


@pytest.mark.parametrize("case", LENGTH_RESNET_CASES)
def test_length_after_resnet(case: LengthResNetCases):
    speaker_encoder = SpeakerEncoder(256, 256, 10, case.num_resnet_blocks)
    batch = torch.randn((5, 256 * 3, case.length))
    x = speaker_encoder.conv1(speaker_encoder.channel_layer_norm(batch))
    x = speaker_encoder.conv2(speaker_encoder.resnet_blocks(x))

    assert x.shape[-1] == speaker_encoder._length_after_resnet(case.length)
