import struct
import re
from typing import List, Optional

import torch

import ppgs


###############################################################################
# PPGs editing API
###############################################################################


def reallocate(
    ppg: torch.Tensor,
    source: str,
    target: str,
    value: Optional[float] = None
) -> torch.Tensor:
    """Reallocate probability from source phoneme to target phoneme

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        source
            Source phoneme
        target
            Target phoneme
        value
            Max amount to reallocate. If None, reallocates all probability.

    Returns
        Edited PPG
    """
    # Get indices corresponding to source and target
    source_index = ppgs.PHONEMES.index(source)
    target_index = ppgs.PHONEMES.index(target)

    # Update source probabilities
    if value:
        reallocation_probability = min(ppg[source_index], value)
        ppg[source_index] = max(0, ppg[source_index] - value)
    else:
        reallocation_probability = ppg[source_index]
        ppg[source_index] = 0.

    # Update target probabilities
    ppg[target_index] += reallocation_probability

    return ppg


def regex(
    ppg: torch.Tensor,
    source_phonemes: List[str],
    target_phonemes: List[str]
) -> torch.Tensor:
    """Regex match and replace (via swap) for phoneme sequences

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        source_phonemes
            Source phoneme sequence
        target_phonemes
            Target phoneme sequence

    Returns
        Edited PPG
    """
    source_indices = [ppgs.PHONEMES.index(phone) for phone in source_phonemes]
    target_indices = [ppgs.PHONEMES.index(phone) for phone in target_phonemes]

    # TODO - prohibits non-one-to-one
    assert len(source_indices) == len(target_indices)

    # Decode to phoneme indices using argmax
    indices = ppg.argmax(dim=1)
    unique_indices, inverse = torch.unique_consecutive(
        indices,
        return_inverse=True)

    # Regex search for source matches
    pattern = re.escape(struct.pack('b'*len(source_indices), *source_indices))
    string = struct.pack('b'*len(unique_indices), *unique_indices)
    match_indices = [match.span()[0] for match in re.finditer(pattern, string)]

    # Swap matched probability sequences with target phoneme sequence
    for i in range(0, len(source_phonemes)):
        slicing = torch.isin(inverse, match_indices + i)
        temporary = ppg[target_indices[i], slicing].clone()
        ppg[target_indices[i], slicing] = ppg[source_indices[i], slicing]
        ppg[source_indices[i], slicing] = temporary

    return ppg


def shift(ppg: torch.Tensor, phoneme: str, value: float):
    """Shift probability of a phoneme and reallocate proportionally

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        phoneme
            Input phoneme
        value
            Maximal shift amount

    Returns
        Edited PPG
    """
    # Get index of phoneme to shift
    index = ppgs.PHONEMES.index(phoneme)

    # Get residual indices
    residual_indices = torch.tensor([
        i for i in range(len(ppgs.PHONEMES)) if i != index])

    # Per-frame shift value
    if value > 0:
        frame_values = torch.min(1. - ppg[index], value)
    else:
        frame_values = torch.max(ppg[index], value)
    # Update target phoneme
    ppg[index] += frame_values

    # Update residual phonemes
    proportion = torch.softmax(ppg[residual_indices], dim=1)
    ppg[residual_indices] = -frame_values * proportion

    # TEMPORARY
    assert (ppg <= 1.0 & ppg >= 0.).all()

    return ppg


def swap(ppg: torch.Tensor, phonemeA: str, phonemeB: str) -> torch.Tensor:
    """Swap the probabilities of two phonemes

    Arguments
        ppg
            Input PPG            shape=(len(ppg.PHONEMES), frames)
        phonemeA
            Input phoneme A
        phonemeB
            Input phoneme B

    Returns
        Edited PPG
    """
    # Get indices of phoneme probabilities to swap
    indexA = ppgs.PHONEMES.index(phonemeA)
    indexB = ppgs.PHONEMES.index(phonemeB)

    # Swap probabilities
    # TODO - clone?
    tmp = ppg[indexA]
    ppg[indexA] = ppg[indexB]
    ppg[indexB] = tmp

    return ppg
