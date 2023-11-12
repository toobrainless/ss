import json
import logging
from pathlib import Path

from tqdm import tqdm

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        self._data_dir = Path(data_dir)
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    @staticmethod
    def _update_target_dict(target_dict, ref_speaker_id):
        if ref_speaker_id not in target_dict:
            ref_target = len(target_dict)
            target_dict[ref_speaker_id] = ref_target
        else:
            ref_target = target_dict[ref_speaker_id]
        return ref_target

    def _create_index(self):
        index = []
        mix_dir = self._data_dir / "mix"
        speaker_id_target = {}
        for mix_path in tqdm(list(mix_dir.glob("*.wav"))):
            ref_path = (
                mix_path.parent.parent
                / "refs"
                / f"{mix_path.stem.split('-')[0]}-ref.wav"
            )
            if not ref_path.exists():
                print(f"Ref path {ref_path} does not exist")
                continue
            target_path = (
                mix_path.parent.parent
                / "targets"
                / f"{mix_path.stem.split('-')[0]}-target.wav"
            )
            if not target_path.exists():
                print(f"Target path {target_path} does not exist")
                continue
            ref_speaker_id = int(mix_path.stem.split("_")[0])

            ref_target = self._update_target_dict(speaker_id_target, ref_speaker_id)

            index.append(
                {
                    "mix_path": str(mix_path.absolute()),
                    "ref_path": str(ref_path.absolute()),
                    "target_path": str(target_path.absolute()),
                    "ref_speaker_id": int(mix_path.stem.split("_")[0]),
                    "ref_target": ref_target,
                }
            )

        return index
