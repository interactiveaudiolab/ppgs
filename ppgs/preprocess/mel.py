import multiprocessing as mp
import os

import torch

import ppgs


###############################################################################
# Spectrogram computation
###############################################################################


def from_audios(audio, lengths, sample_rate=ppgs.SAMPLE_RATE, gpu=None):
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'
    audio = audio.to(device)
    spec = ppgs.preprocess.spectrogram.from_audios(audio, lengths)
    melspec = linear_to_mel(spec)
    return melspec.to(torch.float16)


def from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, gpu=None):
    with torch.autocast('cuda' if gpu is not None else 'cpu'):
        if audio.dim() == 2:
            audio = audio.unsqueeze(dim=0)
        return from_audios(
            audio,
            lengths=audio.shape[-1],
            sample_rate=sample_rate,
            gpu=gpu)


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
    import librosa

    # Create mel basis
    if not hasattr(linear_to_mel, 'mel_basis'):
        basis = librosa.filters.mel(
            sr=ppgs.SAMPLE_RATE,
            n_fft=ppgs.NUM_FFT,
            n_mels=ppgs.NUM_MELS)
        basis = torch.from_numpy(basis)
        basis = basis.to(spectrogram.device)
        linear_to_mel.basis = basis

    # Convert to mels
    original_dtype = spectrogram.dtype
    melspectrogram = torch.matmul(
        linear_to_mel.basis,
        spectrogram.to(torch.float))

    # Apply dynamic range compression
    return torch.log(torch.clamp(melspectrogram, min=1e-5)).to(original_dtype)
