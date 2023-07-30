import multiprocessing as mp
import os

import torch
from tqdm.contrib.concurrent import process_map

import ppgs
from ppgs.data.disk import stop_if_disk_full

###############################################################################
# Audio unfolding
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
            from_features.model.load_state_dict(torch.load(ppgs.CHECKPOINT_DIR / 'unfold.pt')['model'])
        from_features.model.to(features.device)
    return from_features.model(features, new_lengths)

def from_audios(
    audio,
    lengths,
    sample_rate=None,
    config=None,
    gpu=None
):
    return from_audio(audio)

def from_audios(
    audio,
    lengths,
    gpu=None):
    """compute unfold representation from audios"""
    expected_length = audio.shape[-1] // ppgs.HOPSIZE
    size = (ppgs.NUM_FFT - ppgs.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    unfolded = torch.nn.functional.unfold(
        audio[:, None],
        (1, ppgs.WINDOW_SIZE),
        stride=(1, ppgs.HOPSIZE)
    ).squeeze()
    assert unfolded.shape[-1] == expected_length

    return unfolded.to(torch.float16)

def from_audio(audio):
    """Compute unfold representation from audio"""
    # Pad audio
    expected_length = audio.shape[-1] // ppgs.HOPSIZE
    size = (ppgs.NUM_FFT - ppgs.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    # Compute stft
    unfolded = torch.nn.functional.unfold(
        audio[:, None, None],
        (1, ppgs.WINDOW_SIZE),
        stride=(1, ppgs.HOPSIZE)
    ).squeeze()
    assert unfolded.shape[-1] == expected_length

    # Compute magnitude
    return unfolded.to(torch.float16)

def from_file(audio_file):
    """Compute unfold from audio file"""
    audio = ppgs.load.audio(audio_file)
    return from_audio(audio)

def from_file_to_file(audio_file, output_file):
    """Compute unfold from audio file and save to disk"""
    stop_if_disk_full()
    output = from_file(audio_file)
    torch.save(output, output_file)

def from_files_to_files(audio_files, output_files):
    """Compute unfold from audio files and save to disk"""
    process_map(from_file_to_file, audio_files, output_files, max_workers=16, chunksize=512)