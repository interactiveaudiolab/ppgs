import ppgs
import pypar
import tqdm

def charsiu(features=None, gpu=None):
    """Perform preprocessing for charsiu dataset"""

    data_dir = ppgs.DATA_DIR / 'charsiu'
    wav_dir = data_dir / 'wav'
    textgrid_dir = data_dir / 'textgrid'
    output_dir = ppgs.SOURCES_DIR / 'charsiu'

    output_dir.mkdir(exist_ok=True, parents=True)

    audio_files = list(wav_dir.glob('*.wav'))

    with ppgs.data.chdir(output_dir):

        if 'phonemes' in features: #convert textgrid and transfer
            raise NotImplementedError('phoneme preprocessing for charsiu not fully implemented')
            textgrid_files = textgrid_dir.glob('*.textgrid')
            iterator = tqdm.tqdm(
                textgrid_files,
                desc="Converting textgrid phone dialect for charsiu dataset",
                total=len(textgrid_files),
                dynamic_ncols=True
            )
            for textgrid_file in iterator:
                alignment = pypar.Alignment(textgrid_file)

        if 'wav' in features: #copy wav files
            raise NotImplementedError('wav preprocessing for charsiu not fully implemented')

        if 'ppg' in features: #compute ppgs
            ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
            ppgs.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu=gpu
            )

        if 'w2v2' in features: #compute w2v2 latents
            audio_files = audio_files
            w2v2_files = [f'{file.stem}-w2v2.pt' for file in audio_files]
            ppgs.preprocess.w2v2.from_files_to_files(
                audio_files,
                w2v2_files,
                gpu=gpu
            )
