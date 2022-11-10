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

    def __init__(self, name, partition, representation='ppg'):
        self.representation = representation
        self.cache = ppgs.CACHE_DIR / name
        self.stems = ppgs.load.partition(name)[partition]

        #prep phoneme mapping in pyfoal
        pyfoal.convert.phoneme_to_index.map = ppgs.PHONEME_TO_INDEX_MAPPING

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load ppgs
        input_ppgs = torch.load(self.cache / f'{stem}-{self.representation}.pt')

        # Also load audio for evaluation purposes
        audio = torchaudio.load(self.cache / f'{stem}.wav')

        # Load alignment
        # Assumes alignment is saved as a textgrid file, but
        # pypar can also handle json and mfa
        alignment = pypar.Alignment(self.cache / f'{stem}.textgrid')

        # This assumes that frame zero of ppgs is centered on sampled zero,
        # frame one is centered on sample ppgs.HOPSIZE, frame two is centered
        # on sample 2 * ppgs.HOPSIZE, etc. Adjust accordingly.
        hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
        #TODO investigate -1
        # times = np.arange(input_ppgs.shape[-1]) * hopsize
        # if not input_ppgs.shape[-1]*hopsize <= alignment.duration():
        #     raise ValueError(input_ppgs.shape[-1]*hopsize, alignment.duration())
        
        # times = np.arange(0, input_ppgs.shape[-1]*hopsize, hopsize)
        # if not input_ppgs.shape[-1] == times.shape[0]:
        #     raise ValueError(input_ppgs.shape[-1], times.shape[0], list(times), alignment.duration(), input_ppgs.shape[-1]*hopsize, hopsize)

        # samples = np.arange(0, audio[0].shape[-1], ppgs.HOPSIZE, dtype=np.longdouble)
        # times = samples / ppgs.SAMPLE_RATE

        percentages = np.arange(0, 1, ppgs.HOPSIZE / audio[0].shape[-1], dtype=np.longdouble)
        times = percentages * alignment.duration()


        #Fix last time to be within duration
        # if times[-1] > alignment.duration():
        #     times[-1] = alignment.duration() - 1e-6
        # if stem == 'cmu_us_bdl_arctic/arctic_a0481':
        #     print(times[-1], alignment.duration())
        #     print(list(times))
        #     print(times[-1] <= times[-2])

        # Convert alignment to framewise indices
        try:
            indices, word_breaks = pyfoal.convert.alignment_to_indices(
                alignment,
                hopsize=hopsize,
                return_word_breaks=True,
                times=times)
        except ValueError as e:
            raise ValueError(f'error processing alignment for stem {stem} with error: {e}')
        indices = torch.tensor(indices, dtype=torch.long)

        return input_ppgs, indices, alignment, word_breaks, audio, stem

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
