import ppgs
import pypar
import tqdm
from shutil import copy as cp
from torch import save, float16
import multiprocessing as mp
import sys
from ppgs.data import preserve_free_space, stop_if_disk_full

# @preserve_free_space
def save_masked(tensor, file, length):
    sub_tensor = tensor[:, :length].clone()
    save(sub_tensor, file)

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
    else:
        dataloader = ppgs.preprocess.accel.loader('charsiu', num_workers=num_workers)
        feature_processors = [ppgs.REPRESENTATION_MAP[f] for f in features]
        iterator = tqdm.tqdm(
            dataloader,
            desc=f'preprocessing charsiu dataset for features {features}',
            total=len(dataloader),
            dynamic_ncols=True
        )
        with mp.get_context('spawn').Pool(8) as pool:
            for audios, audio_files, lengths in iterator:
                for feature, feature_processor in zip(features, feature_processors):
                    outputs = feature_processor.from_audios(audios, lengths, gpu=gpu).cpu().to(float16)
                    assert str(outputs.device) == 'cpu', f'"{outputs.device}"'
                    new_lengths = [length // ppgs.HOPSIZE for length in lengths]
                    filenames = [output_dir / f'{audio_file.stem}-{feature}.pt' for audio_file in audio_files]
                    pool.starmap_async(save_masked, zip(outputs, filenames, new_lengths))
                stop_if_disk_full()
            pool.close()
            pool.join()
