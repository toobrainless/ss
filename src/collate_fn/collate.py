import logging
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


# TODO write callable class
def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    for audio_type in ["ref", "target", "mix"]:
        result_batch[f"{audio_type}_duration"] = []
        result_batch[f"{audio_type}_path"] = []
        result_batch[f"{audio_type}_audio"] = []
        result_batch[f"{audio_type}_length"] = []

    result_batch["ref_speaker_id"] = []
    result_batch["ref_target"] = []

    for item in dataset_items:
        for audio_type in ["ref", "target", "mix"]:
            result_batch[f"{audio_type}_duration"].append(
                item[f"{audio_type}_duration"]
            )
            result_batch[f"{audio_type}_audio"].append(item[f"{audio_type}_audio"][0])
            result_batch[f"{audio_type}_path"].append(item[f"{audio_type}_path"])
            result_batch[f"{audio_type}_length"].append(item[f"{audio_type}_length"])
        result_batch["ref_speaker_id"].append(item["ref_speaker_id"])
        result_batch["ref_target"].append(item["ref_target"])

    for audio_type in ["ref", "target", "mix"]:
        result_batch[f"{audio_type}_duration"] = torch.tensor(
            result_batch[f"{audio_type}_duration"]
        )
        result_batch[f"{audio_type}_length"] = torch.tensor(
            result_batch[f"{audio_type}_length"]
        )
        result_batch[f"{audio_type}_audio"] = pad_sequence(
            result_batch[f"{audio_type}_audio"], batch_first=True
        )
    result_batch["ref_speaker_id"] = torch.tensor(result_batch["ref_speaker_id"])
    result_batch["ref_target"] = torch.tensor(result_batch["ref_target"])

    return result_batch
