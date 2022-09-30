"""core.py - data preprocessing"""


import ppgs
from os import listdir, makedirs
from os.path import join, isfile, isdir

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'phonemes', 'spectrogram']


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

            audio_files = sorted(list(audio_dir.glob('*.wav')))
            phoneme_files = sorted(list(phoneme_dir.glob('*.csv')))

            sentences_file = speaker_dir / 'sentences.csv'

            from_files_to_files(
                speaker_output_dir, 
                audio_files, 
                phoneme_files, 
                sentences_file=sentences_file, 
                features=features, gpu=gpu)
            break



def from_files_to_files(
    output_directory,
    audio_files,
    phone_files,
    sentences_file=None,
    features=ALL_FEATURES,
    gpu=None):
    """Preprocess from files"""
    # Change directory
    with ppgs.data.chdir(output_directory):

        if 'phonemes' in features:
            alignment_files = [f'{file.stem}.TextGrid' for file in phone_files]
            # print(phone_files[:10])
            # print(sentences_file)
            ppgs.preprocess.words.from_files_to_files(
                phone_files,
                alignment_files,
                sentences_file
            )

        # Preprocess spectrograms
        # if 'spectrogram' in features:
        #     spectrogram_files = [
        #         f'{file.stem}-spectrogram.pt' for file in audio_files]
        #     ppgs.preprocess.spectrogram.from_files_to_files(
        #         audio_files,
        #         spectrogram_files)

        # Preprocess phonetic posteriorgrams
        if 'ppg' in features:
            ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
            ppgs.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu)