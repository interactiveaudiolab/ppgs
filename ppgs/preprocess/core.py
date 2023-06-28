"""core.py - data preprocessing"""


import ppgs
from os import listdir, makedirs
from os.path import join, isdir
from shutil import copy as cp
from ppgs.preprocess.charsiu import charsiu
from ppgs.notify import notify_on_finish
from ppgs.preprocess.accel import multiprocessed_preprocess
from tqdm import tqdm

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['phonemes', 'wav', 'w2v2fs', 'bottleneck', 'w2v2fb', 'spectrogram', 'mel', 'unfold', 'encodec']


###############################################################################
# Preprocess
###############################################################################

@notify_on_finish('preprocessing')
def datasets(datasets, features=ALL_FEATURES, gpu=None, use_cached_inputs=False, num_workers=-1):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = ppgs.DATA_DIR / dataset if not use_cached_inputs else ppgs.CACHE_DIR / dataset
        output_directory = ppgs.CACHE_DIR / dataset

        if dataset == 'charsiu':
            charsiu(input_directory, output_directory, features=features, num_workers=num_workers, gpu=gpu)
            continue

        if num_workers == -1:

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
        else:
            print('using multiprocessed preprocessing')
            multiprocessed_preprocess(dataset, None, features, num_workers, gpu)



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
            iterator = tqdm(
                audio_files,
                desc=f'copying audio files for speaker',
                total=len(audio_files),
                dynamic_ncols=True
            )
            for file in iterator:
                cp(file, output_directory / file.name)

        # Preprocess phonetic posteriorgrams
        if 'bottleneck' in features:
            ppg_files = [f'{file.stem}-bottleneck.pt' for file in audio_files]
            ppgs.preprocess.bottleneck.from_files_to_files(
                audio_files,
                ppg_files,
                gpu
            )

        # Preprocess wav2vec2-fs latents
        if 'w2v2fs' in features:
            w2v2fs_files = [f'{file.stem}-w2v2fs.pt' for file in audio_files]
            ppgs.preprocess.w2v2fs.from_files_to_files(
                audio_files,
                w2v2fs_files,
                gpu
            )

        # Preprocess wav2vec2-fb latents
        if 'w2v2fb' in features:
            w2v2fb_files = [f'{file.stem}-w2v2fb.pt' for file in audio_files]
            ppgs.preprocess.w2v2fb.from_files_to_files(
                audio_files,
                w2v2fb_files,
                gpu
            )

        if 'mel' in features:
            mel_files = [f'{file.stem}-mel.pt' for file in audio_files]
            ppgs.preprocess.spectrogram.from_files_to_files(audio_files, mel_files, mels=True)

        if 'spectrogram' in features:
            spectrogram_files = [f'{file.stem}-spectrogram.pt' for file in audio_files]
            ppgs.preprocess.spectrogram.from_files_to_files(audio_files, spectrogram_files, mels=False)

def from_audio(audio, representation=None, sample_rate=ppgs.SAMPLE_RATE, config=None, gpu=None):
    """Preprocess audio using given or configured representation"""

    #Cache model/function
    if representation is None:
        representation = ppgs.REPRESENTATION
    try:
        representation_module = ppgs.REPRESENTATION_MAP[representation]
    except KeyError:
        raise ValueError(f'given representation "{representation}" does not exist')
    if not hasattr(from_audio, representation):
        setattr(from_audio, representation, representation_module.from_audio)

    #Compute representation
    return getattr(from_audio, representation)(
        audio, 
        sample_rate=sample_rate,
        config=config,
        gpu=gpu
    )