import torch
import torchutil
from huggingface_hub import hf_hub_download

import ppgs


###############################################################################
# Constants
###############################################################################


# Configuration
CONFIG_FILE = ppgs.ASSETS_DIR / 'configs' / 'bottleneck.yaml'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

# Window size of the model
WINDOW_SIZE = 1024


###############################################################################
# Preprocess ASR bottleneck features
###############################################################################


def from_audios(
    audio,
    lengths,
    sample_rate=ppgs.SAMPLE_RATE,
    config=CONFIG_FILE,
    gpu=None):
    """Compute ASR bottleneck features from audio"""
    with torch.no_grad():
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audios, 'model'):
            conformer_checkpoint_file = hf_hub_download(
                repo_id='CameronChurchwell/ppg_conformer_model',
                filename='24epoch.pth')
            from_audios.model = ppgs.preprocess.bottleneck.conformer_ppg_model.build_ppg_model.load_ppg_model(
                config,
                conformer_checkpoint_file,
                device)

        # Maybe resample
        audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE)

        # Setup features
        pad = WINDOW_SIZE//2 - ppgs.HOPSIZE//2
        lengths = lengths + 2*pad
        audio = torch.nn.functional.pad(audio, (pad, pad))
        audio = audio.to(device)
        audio = audio.squeeze(dim=1)

        # Infer Bottleneck PPGs
        output = from_audios.model(audio, lengths).transpose(1, 2)
        return output.to(torch.float16)


def from_audio(
    audio,
    sample_rate=ppgs.SAMPLE_RATE,
    config=CONFIG_FILE,
    gpu=None):
    """Compute Bottleneck PPGs from audio"""
    with torch.no_grad():
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audio, 'model'):
            conformer_checkpoint_file = hf_hub_download(repo_id='CameronChurchwell/ppg_conformer_model', filename='24epoch.pth')
            from_audio.model = ppgs.preprocess.bottleneck.conformer_ppg_model.build_ppg_model.load_ppg_model(
                config,
                conformer_checkpoint_file,
                device)

        # Maybe resample
        audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE)

        # Setup features
        audio = audio.to(device)
        pad = WINDOW_SIZE // 2 - ppgs.HOPSIZE // 2
        length = (
            torch.tensor([audio.shape[-1]], dtype=torch.long, device=device) +
            2 * pad)
        audio = torch.nn.functional.pad(audio, (pad, pad))

        # Infer Bottleneck PPGs
        with torch.no_grad():
            return from_audio.model(audio, length)[0].T


def from_file(audio_file, gpu=None):
    """Compute Bottleneck PPGs from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute Bottleneck PPGs from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute Bottleneck PPGs from audio files and save to disk"""
    for audio_file, output_file in torchutil.iterator(
        zip(audio_files, output_files),
        'Extracting PPGs',
        total=len(audio_files)
    ):
        from_file_to_file(audio_file, output_file, gpu)
