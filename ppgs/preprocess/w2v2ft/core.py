import torch
import tqdm
from transformers.utils import logging

import ppgs
from ppgs.model.transformer import mask_from_lengths

###############################################################################
# Constants
###############################################################################

# W2V2 FB pretrained model config name
W2V2FB_CONFIG = "facebook/wav2vec2-base"

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400
HOP_SIZE = 160


###############################################################################
# Phonetic posteriorgram
###############################################################################

logging.set_verbosity_error()

def from_audios(
    audio,
    lengths,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2FB latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze(dim=1)
    pad = WINDOW_SIZE//2 - HOP_SIZE//2
    padded_audio = torch.nn.functional.pad(audio, (pad, pad))
    return padded_audio

def from_audio(
    audio,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute audio tensor latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze(dim=1)
    pad = WINDOW_SIZE//2 - HOP_SIZE//2
    padded_audio = torch.nn.functional.pad(audio, (pad, pad))
    return padded_audio


def from_file(audio_file, gpu=None):
    """Compute audio tensor from file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute audio tensors from file and save to file"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute audio tensors from files and save to files"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting W2V2FT latents',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)
