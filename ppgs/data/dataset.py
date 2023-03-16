"""dataset.py - data loading"""

import contextlib
import warnings
import numpy as np
import pyfoal
import pypar
import torch
import torchaudio

import ppgs


###############################################################################
# Dataset
###############################################################################

@contextlib.contextmanager
def ppgs_phoneme_list():
    try:
        setattr(pyfoal.convert.phoneme_to_index, 'map', ppgs.PHONEME_TO_INDEX_MAPPING)
        yield
    finally:
        delattr(pyfoal.convert.phoneme_to_index, 'map')

class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition, representation='senone', reduced_features=False):
        self.representation = representation
        self.metadata = Metadata(name, partition)
        self.cache = ppgs.CACHE_DIR / name
        self.stems = ppgs.load.partition(name)[partition]
        self.reduced_features = reduced_features

        #calculate window size based on representation #TODO consider removing
        # self.WINDOW_SIZE = getattr(ppgs.preprocess, representation).WINDOW_SIZE

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # Load ppgs
        input_ppgs = torch.load(self.cache / f'{stem}-{self.representation}.pt')

        if self.reduced_features: #load only features necessary for training
            audio_num_samples = torchaudio.info(self.cache / f'{stem}.wav').num_frames
        else: #load all features
            audio = torchaudio.load(self.cache / f'{stem}.wav') #TODO refactor out?
            audio_num_samples = audio[0].shape[-1]

        num_frames = audio_num_samples//ppgs.HOPSIZE


        # Load alignment
        alignment = pypar.Alignment(self.cache / f'{stem}.textgrid')


        # This assumes that frame zero of ppgs is centered on sampled zero,
        # frame one is centered on sample ppgs.HOPSIZE, frame two is centered
        # on sample 2 * ppgs.HOPSIZE, etc. Adjust accordingly.
        hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
        times = np.linspace(hopsize/2, (num_frames-1)*hopsize+hopsize/2, num_frames)
        times[-1] = alignment.duration()

        if times.shape[0] != input_ppgs.shape[-1]:
            raise ValueError('Non-matching data shapes!')

        # Convert alignment to framewise indices
        try:
            #prep phoneme mapping in pyfoal
            with ppgs_phoneme_list():
                indices = pyfoal.convert.alignment_to_indices(
                    alignment,
                    hopsize=hopsize,
                    return_word_breaks=not self.reduced_features,
                    times=times)
                if not self.reduced_features:
                    indices, word_breaks = indices
        except ValueError as e:
            raise ValueError(f'error processing alignment for stem {stem} with error: {e}')
        indices = torch.tensor(indices, dtype=torch.long)

        if self.reduced_features:
            return input_ppgs, indices

        return input_ppgs, indices, alignment, word_breaks, audio, stem

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // ppgs.BUCKETS

        # Get indices in order of length
        lengths = []
        # for i in range(len(self)):
        #     index, dataset = self.get_dataset(i)
        #     lengths.append(dataset.lengths[index])
        lengths = self.metadata.lengths
        indices = np.argsort(lengths)

        # Split into buckets based on length
        buckets = [indices[i:i + size] for i in range(0, len(self), size)]

        # Add max length of each bucket
        buckets = [(lengths[bucket[-1]], bucket) for bucket in buckets]

        return buckets

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)


###############################################################################
# Metadata
###############################################################################

class Metadata:

    def __init__(self, name, partition):
        self.name = name
        self.cache = ppgs.CACHE_DIR / name
        self.stems = ppgs.load.partition(name)[partition]

        print('preparing dataset metadata (operation may be slow)')

        # Store lengths for bucketing
        audio_files = list([
            self.cache / f'{stem}.wav' for stem in self.stems])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.lengths = [torchaudio.info(audio_file).num_frames // ppgs.HOPSIZE for audio_file in audio_files]

    def __len__(self):
        return len(self.stems)