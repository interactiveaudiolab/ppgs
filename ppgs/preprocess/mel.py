import multiprocessing as mp
import os

import torch
import librosa

import ppgs
from . import spectrogram


###############################################################################
# Spectrogram computation
###############################################################################

def from_features(
    features: torch.Tensor,
    new_lengths: torch.Tensor,
    checkpoint=None,
    gpu=0
):
    if not hasattr(from_features, 'model'):
        from_features.model = ppgs.Model()()
        if checkpoint is not None:
            from_features.model.load_state_dict(torch.load(checkpoint)['model'])
        else:
            from_features.model.load_state_dict(torch.load(ppgs.CHECKPOINT_DIR / 'mel.pt')['model'])
        from_features.model.to(features.device)
    return from_features.model(features, new_lengths)

def from_audios(
    audio,
    lengths,
    gpu=None,
):
    spec = spectrogram.from_audios(audio, lengths)
    melspec = linear_to_mel(spec)
    return melspec.to(torch.float16)

def from_audio(audio, mels=False):
    """Compute spectrogram from audio"""
    spec = spectrogram.from_audio(audio)
    melspec = linear_to_mel(spec) if mels else spectrogram

    return melspec.squeeze(0).to(torch.float16)


def from_file(audio_file):
    """Compute spectrogram from audio file"""
    audio = ppgs.load.audio(audio_file)
    return from_audio(audio)


def from_file_to_file(audio_file, output_file):
    """Compute spectrogram from audio file and save to disk"""
    output = from_file(audio_file)
    torch.save(output, output_file)


def from_files_to_files(audio_files, output_files):
    """Compute mel from audio files and save to disk"""
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(from_file_to_file, zip(audio_files, output_files))


###############################################################################
# Utilities
###############################################################################


def linear_to_mel(spectrogram):
    # Create mel basis
    if not hasattr(linear_to_mel, 'mel_basis'):
        basis = librosa.filters.mel(
            sr=ppgs.SAMPLE_RATE,
            n_fft=ppgs.NUM_FFT,
            n_mels=ppgs.NUM_MELS)
        basis = torch.from_numpy(basis)
        basis = basis.to(spectrogram.dtype).to(spectrogram.device)
        linear_to_mel.basis = basis

    # Convert to mels
    melspectrogram = torch.matmul(linear_to_mel.basis, spectrogram)

    # Apply dynamic range compression
    return torch.log(torch.clamp(melspectrogram, min=1e-5))
