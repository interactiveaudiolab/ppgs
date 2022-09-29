"""dataset.py - data loading"""

import numpy as np
import pyfoal
import pypar
import torch
import torchaudio

import ppgs


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition):
        self.cache = ppgs.CACHE_DIR / name
        self.stems = ppgs.load.partition(name)[partition]

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load ppgs
        input_ppgs = torch.load(self.cache / f'{stem}-ppg.pt')

        # This assumes that frame zero of ppgs is centered on sampled zero,
        # frame one is centered on sample ppgs.HOPSIZE, frame two is centered
        # on sample 2 * ppgs.HOPSIZE, etc. Adjust accordingly.
        hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
        times = np.arange(input_ppgs.shape[-1]) * hopsize

        # Load alignment
        # Assumes alignment is saved as a textgrid file, but
        # pypar can also handle json and mfa
        alignment = pypar.Alignment(self.cache / f'{stem}.TextGrid')

        # Convert alignment to framewise indices
        indices = pyfoal.alignment_to_indices(
            alignment,
            hopsize=hopsize,
            return_word_breaks=True,
            times=times)
        indices = torch.tensor(indices, dtype=torch.long)

        # Also load audio for evaluation purposes
        audio = torchaudio.load(self.cache / f'{stem}.wav')

        return input_ppgs, indices, alignment, audio

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
