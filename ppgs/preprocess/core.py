"""core.py - data preprocessing"""


import ppgs
from os import listdir, makedirs
from os.path import join, isfile, isdir

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'text', 'spectrogram']


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

            audio_files = sorted(list(audio_dir.glob('*.wav')))

            from_files_to_files(speaker_output_dir, audio_files)

            
        
def from_files_to_files(
    output_directory,
    audio_files,
    text_files=None,
    features=ALL_FEATURES,
    gpu=None):
    """Preprocess from files"""
    # Change directory
    with ppgs.data.chdir(output_directory):

        # Preprocess phonemes from text
        # if 'phonemes' in features:
        #     phoneme_files = [
        #         f'{file.stem}-text.pt' for file in text_files]
        #     ppgs.preprocess.text.from_files_to_files(
        #         text_files,
        #         phoneme_files)

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

        # Preprocess prosody features
        # if 'prosody' in features:
        #     prefixes = [Path(file.stem) for file in audio_files]
        #     pysodic.from_files_to_files(
        #         audio_files,
        #         prefixes,
        #         text_files,
        #         ppgs.HOPSIZE / ppgs.SAMPLE_RATE,
        #         ppgs.WINDOW_SIZE / ppgs.SAMPLE_RATE,
        #         gpu)


# def prosody(audio, sample_rate=ppgs.SAMPLE_RATE, text=None, gpu=None):
#     """Preprocess prosody from audio to retrieve features for editing"""
#     hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
#     window_size = ppgs.WINDOW_SIZE / ppgs.SAMPLE_RATE

#     # Get prosody features including alignment
#     if text:
#         output = pysodic.from_audio_and_text(
#             audio,
#             sample_rate,
#             text,
#             hopsize,
#             window_size,
#             gpu)

#         # Pitch, loudness and alignment
#         return output[0], output[2], output[5]

#     # Get prosody features without alignment
#     output = pysodic.from_audio(audio, sample_rate, hopsize, window_size, gpu)

#     # Pitch and loudness
#     return output[0], output[2]