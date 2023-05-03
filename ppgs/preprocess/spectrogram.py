import functools
import multiprocessing as mp
import os

import torch
import librosa

import ppgs


###############################################################################
# Spectrogram computation
###############################################################################


def from_audio(audio, mels=False):
    """Compute spectrogram from audio"""
    # Cache hann window
    if (
        not hasattr(from_audio, 'window') or
        from_audio.dtype != audio.dtype or
        from_audio.device != audio.device
    ):
        from_audio.window = torch.hann_window(
            ppgs.WINDOW_SIZE,
            dtype=audio.dtype,
            device=audio.device)
        from_audio.dtype = audio.dtype
        from_audio.device = audio.device

    # Pad audio
    size = (ppgs.NUM_FFT - ppgs.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    # Compute stft
    stft = torch.stft(
        audio.squeeze(1),
        ppgs.NUM_FFT,
        hop_length=ppgs.HOPSIZE,
        window=from_audio.window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True)
    stft = torch.view_as_real(stft)

    # Compute magnitude
    spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-6)

    # Maybe convert to mels
    spectrogram = linear_to_mel(spectrogram) if mels else spectrogram

    return spectrogram.squeeze(0).to(torch.float16)


def from_file(audio_file, mels=False):
    """Compute spectrogram from audio file"""
    audio = ppgs.load.audio(audio_file)
    return from_audio(audio, mels)


def from_file_to_file(audio_file, output_file, mels=False):
    """Compute spectrogram from audio file and save to disk"""
    output = from_file(audio_file, mels)
    torch.save(output, output_file)


def from_files_to_files(audio_files, output_files, mels=False):
    """Compute spectrogram from audio files and save to disk"""
    preprocess_fn = functools.partial(from_file_to_file, mels=mels)
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(preprocess_fn, zip(audio_files, output_files))


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
