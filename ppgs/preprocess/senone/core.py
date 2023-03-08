import torch
import torchaudio
import tqdm

import ppgs

###############################################################################
# Constants
###############################################################################


# PPG model checkpoint file
CHECKPOINT_FILE = ppgs.ASSETS_DIR / 'checkpoints' / 'ppg.pt'

# PPG model configuration
CONFIG_FILE = ppgs.ASSETS_DIR / 'configs' / 'ppg.yaml'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 1024


###############################################################################
# Phonetic posteriorgram
###############################################################################


def from_audio(
    audio,
    sample_rate=ppgs.SAMPLE_RATE,
    config=CONFIG_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    gpu=None):
    """Compute Senone PPGs from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=CONFIG_FILE
    if checkpoint_file is None: checkpoint_file=CHECKPOINT_FILE
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Cache model
    if not hasattr(from_audio, 'model'):
        from_audio.model = ppgs.preprocess.senone.conformer_ppg_model.build_ppg_model.load_ppg_model(
            config,
            checkpoint_file,
            device)

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE)

    # Setup features
    audio = audio.to(device)
    pad = WINDOW_SIZE//2 - ppgs.HOPSIZE//2
    length = torch.tensor([audio.shape[-1]], dtype=torch.long, device=device) + 2*pad #needs to be caluclated prior to padding
    audio = torch.nn.functional.pad(audio, (pad, pad))

    # Infer senone PPGs
    with torch.no_grad():
        return from_audio.model(audio, length)[0].T


def from_file(audio_file, gpu=None):
    """Compute Senone PPGs from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute Senone PPGs from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute Senone PPGs from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting PPGs',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)