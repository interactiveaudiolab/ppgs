"""core.py - data preprocessing"""


import ppgs
from os import listdir, makedirs
from os.path import join, isdir
from shutil import copy as cp

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'phonemes', 'wav']


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets, features=ALL_FEATURES, gpu=None):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = ppgs.DATA_DIR / dataset
        output_directory = ppgs.CACHE_DIR / dataset

        speakers = [speaker for speaker in listdir(input_directory) if isdir(join(input_directory, speaker))]

        for speaker in speakers:
            print('Preprocessing for speaker', speaker, 'in dataset', dataset)
            speaker_dir = input_directory / speaker

            speaker_output_dir = output_directory / speaker
            makedirs(speaker_output_dir, exist_ok=True)

            audio_dir = speaker_dir / 'wav'
            phoneme_dir = speaker_dir / 'lab'
            word_dir = speaker_dir / 'word'

            audio_files = sorted(list(audio_dir.glob('*.wav')))
            phoneme_files = sorted(list(phoneme_dir.glob('*.csv')))

            from_files_to_files(
                speaker_output_dir, 
                audio_files, 
                phoneme_files,
                word_dir,
                features=features, 
                gpu=gpu)



def from_files_to_files(
    output_directory,
    audio_files,
    phone_files,
    word_directory,
    features=ALL_FEATURES,
    gpu=None):
    """Preprocess from files"""
    # Change directory
    with ppgs.data.chdir(output_directory):

        #Copy phoneme files
        if 'phonemes' in features:
            ppgs.preprocess.align.from_files_to_files(
                phone_files,
                word_directory,
                output_directory
            )

        # Preprocess spectrograms
        # if 'spectrogram' in features:
        #     spectrogram_files = [
        #         f'{file.stem}-spectrogram.pt' for file in audio_files]
        #     ppgs.preprocess.spectrogram.from_files_to_files(
        #         audio_files,
        #         spectrogram_files)

        # Copy wav files
        if 'wav' in features:
            for file in audio_files:
                cp(file, output_directory / file.name)

        # Preprocess phonetic posteriorgrams
        if 'ppg' in features:
            ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
            ppgs.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu
            )