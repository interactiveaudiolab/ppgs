import torch
import torchutil
import transformers

import ppgs

# Turn off logging
transformers.utils.logging.set_verbosity_error()


###############################################################################
# Constants
###############################################################################


# W2V2 FB pretrained model config name
W2V2FB_CONFIG = 'facebook/wav2vec2-base'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400
HOP_SIZE = 160


###############################################################################
# Preprocess fine-tuned wav2vec 2.0 features
###############################################################################


def from_audios(
    audio,
    lengths,
    sample_rate=ppgs.SAMPLE_RATE,
    config=None,
    gpu=None):
    """Compute W2V2FB latents from audio"""
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze(dim=1)
    pad = WINDOW_SIZE // 2 - HOP_SIZE // 2
    return torch.nn.functional.pad(audio, (pad, pad))


def from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, config=None, gpu=None):
    """Compute audio tensor latents from audio"""
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze(dim=1)
    pad = WINDOW_SIZE // 2 - HOP_SIZE // 2
    return torch.nn.functional.pad(audio, (pad, pad))


def from_file(audio_file, gpu=None):
    """Compute audio tensor from file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute audio tensors from file and save to file"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute audio tensors from files and save to files"""
    for audio_file, output_file in torchutil.iterator(
        zip(audio_files, output_files),
        'Extracting W2V2FT latents',
        total=len(audio_files)
    ):
        from_file_to_file(audio_file, output_file, gpu)
