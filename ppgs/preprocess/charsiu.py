import ppgs
import pypar
import tqdm
from shutil import copy as cp
import multiprocessing as mp
from .accel import multiprocessed_preprocess

def charsiu(input_dir, output_dir, features=None, num_workers=-1, gpu=None):
    """Perform preprocessing for charsiu dataset"""

    print('input_dir:', input_dir)
    print('output_dir:', output_dir)

    wav_dir = input_dir / 'wav'
    if not wav_dir.exists():
        wav_dir = input_dir
    textgrid_dir = input_dir / 'textgrid'
    if not textgrid_dir.exists():
        textgrid_dir = input_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    if num_workers == -1:

        audio_files = list(wav_dir.glob('*.wav'))

        with ppgs.data.chdir(output_dir):

            if 'phonemes' in features: #convert textgrid and transfer
                # raise NotImplementedError('phoneme preprocessing for charsiu not fully implemented')
                textgrid_files = list(textgrid_dir.glob('*.textgrid')) + list(textgrid_dir.glob('*.TextGrid'))
                iterator = tqdm.tqdm(
                    textgrid_files,
                    desc="Converting textgrid phone dialect for charsiu dataset",
                    total=len(textgrid_files),
                    dynamic_ncols=True
                )
                for textgrid_file in iterator:
                    alignment = pypar.Alignment(textgrid_file)
                    for word in alignment._words:
                        if word.word == '[SIL]':
                            word.word = 'sp'
                        for phoneme in word.phonemes:
                            if phoneme.phoneme == '[SIL]':
                                phoneme.phoneme = 'sil'
                            else:
                                phoneme.phoneme = phoneme.phoneme.lower()
                    alignment.save(textgrid_file.stem + '.textgrid')

            if 'wav' in features: #copy wav files
                iterator = tqdm.tqdm(
                    audio_files,
                    desc="copying audio files",
                    total=len(audio_files),
                    dynamic_ncols=True
                )
                for audio_file in iterator:
                    cp(audio_file, audio_file.name)

            if 'bottleneck' in features: #compute ppgs
                ppg_files = [f'{file.stem}-bottleneck.pt' for file in audio_files]
                ppgs.preprocess.bottleneck.from_files_to_files(
                    audio_files,
                    ppg_files,
                    gpu=gpu
                )

            if 'w2v2fs' in features: #compute w2v2fs latents
                audio_files = audio_files
                w2v2fs_files = [f'{file.stem}-w2v2fs.pt' for file in audio_files]
                ppgs.preprocess.w2v2fs.from_files_to_files(
                    audio_files,
                    w2v2fs_files,
                    gpu=gpu
                )

            if 'w2v2fb' in features: #compute w2v2fb latents
                audio_files = audio_files
                w2v2fb_files = [f'{file.stem}-w2v2fb.pt' for file in audio_files]
                ppgs.preprocess.w2v2fb.from_files_to_files(
                    audio_files,
                    w2v2fb_files,
                    gpu=gpu
                )

            if 'mel' in features:
                mel_files = [f'{file.stem}-mel.pt' for file in audio_files]
                ppgs.preprocess.spectrogram.from_files_to_files(audio_files, mel_files, mels=True)

            if 'spectrogram' in features:
                spectrogram_files = [f'{file.stem}-spectrogram.pt' for file in audio_files]
                ppgs.preprocess.spectrogram.from_files_to_files(audio_files, spectrogram_files, mels=False)

            if 'unfold' in features:
                unfold_files = [f'{file.stem}-unfold.pt' for file in audio_files]
                ppgs.preprocess.unfold.from_files_to_files(audio_files, unfold_files)
    else:
        multiprocessed_preprocess('charsiu', output_dir, features, num_workers, gpu)
