import os

import click

from src.utils.mixture_generator import LibriSpeechSpeakerFiles, MixtureGenerator


@click.command()
@click.option(
    "--path_train",
    "-pt",
    default="data/datasets/librispeech/train-clean-360",
    help="Path to train dataset",
)
@click.option(
    "--path_val",
    "-pv",
    default="data/datasets/librispeech/dev-clean",
    help="Path to validation dataset",
)
@click.option(
    "--path_mixtures_train",
    "-pmt",
    default="train_mixtures",
    help="Path to destination train folder",
)
@click.option(
    "--path_mixtures_val",
    "-pmv",
    default="val_mixtures",
    help="Path to destination validation folder",
)
@click.option(
    "--nfiles_train",
    "-nft",
    default=10000,
    help="Number of files to generate",
)
@click.option(
    "--nfiles_val",
    "-nfv",
    default=1000,
    help="Number of files to generate",
)
@click.option(
    "--num_workers",
    "-nw",
    default=2,
    help="Number of workers to use",
)
@click.option(
    "--num_speakers",
    "-ns",
    default=100,
    help="Number of speakers in train dataset",
)
def main(
    path_train,
    path_val,
    path_mixtures_train,
    path_mixtures_val,
    nfiles_train,
    nfiles_val,
    num_workers,
    num_speakers,
):
    speakersTrain = [el.name for el in os.scandir(path_train)][:num_speakers]
    speakersVal = [el.name for el in os.scandir(path_val)]

    speakers_files_train = [
        LibriSpeechSpeakerFiles(i, path_train, audioTemplate="*.flac")
        for i in speakersTrain
    ]
    speakers_files_val = [
        LibriSpeechSpeakerFiles(i, path_val, audioTemplate="*.flac")
        for i in speakersVal
    ]

    mixer_train = MixtureGenerator(
        speakers_files_train, path_mixtures_train, nfiles=nfiles_train, test=False
    )

    mixer_val = MixtureGenerator(
        speakers_files_val, path_mixtures_val, nfiles=nfiles_val, test=True
    )

    mixer_train.generate_mixes(
        snr_levels=[0],
        num_workers=num_workers,
        update_steps=100,
        trim_db=None,
        vad_db=None,
        audioLen=3,
    )

    mixer_val.generate_mixes(
        snr_levels=[0],
        num_workers=num_workers,
        update_steps=100,
        trim_db=None,
        vad_db=None,
        audioLen=3,
    )


if __name__ == "__main__":
    main()
