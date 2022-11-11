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
        num_frames = audio[0].shape[-1]//ppgs.HOPSIZE

        # Pad audio
        pad = ppgs.WINDOW_SIZE//2 - ppgs.HOPSIZE//2
        audio = torch.nn.functional.pad(audio[0], (pad, pad))

        # Load alignment
        # Assumes alignment is saved as a textgrid file, but
        # pypar can also handle json and mfa
        alignment = pypar.Alignment(self.cache / f'{stem}.textgrid')

        # This assumes that frame zero of ppgs is centered on sampled zero,
        # frame one is centered on sample ppgs.HOPSIZE, frame two is centered
        # on sample 2 * ppgs.HOPSIZE, etc. Adjust accordingly.
        hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
        times = np.linspace(hopsize/2, (num_frames-1)*hopsize+hopsize/2, num_frames)
        times[-1] = alignment.duration()

        if times.shape[0] != input_ppgs.shape[-1]:
            import pdb; pdb.set_trace()

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
