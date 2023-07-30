"""dataset.py - data loading"""

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pyfoal
import pypar
import torch
import torchaudio

import ppgs

###############################################################################
# Metadata
###############################################################################

class Metadata:

    def __init__(self, sources, partition=None, overwrite_cache=False):
        if isinstance(sources, str):
            self.name = sources
            self.cache = ppgs.CACHE_DIR / self.name
            partition_dict = ppgs.load.partition(self.name)
            if partition is not None:
                self.stems = partition_dict[partition]
            else:
                self.stems = sum(partition_dict.values(), start=[])
            self.audio_files = [self.cache / (stem + '.wav') for stem in self.stems]
            metadata_file = self.cache / f'{partition}-metadata.json'
            if overwrite_cache and metadata_file.exists():
                os.remove(metadata_file)
            if metadata_file.exists():
                print('using cached metadata')
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                print('generating metadata from scratch')
                metadata = {}
        else:
            self.name = '<list of files>'
            self.audio_files = sources
            self.stems = [Path(file).stem for file in self.audio_files]
            self.cache = None
            metadata = {}
        print('preparing dataset metadata (operation may be slow)')
        self.lengths = []
        for stem, audio_file in zip(self.stems, self.audio_files):
            try:
                self.lengths.append(metadata[stem])
            except KeyError:
                length = torchaudio.info(audio_file).num_frames // ppgs.HOPSIZE
                metadata[stem] = length
                self.lengths.append(length)
        if self.cache is not None:
            with open(metadata_file, 'w+') as f:
                json.dump(metadata, f)

    def __len__(self):
        return len(self.stems)

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
        features - list
            The features to load through __getitem__
    """

    def __init__(self, name, partition=None, features=['wav']):
        assert len(features) > 0, "need to pass at least one feature"
        self.features = features
        self.metadata = Metadata(name, partition=partition)
        self.cache = self.metadata.cache
        self.stems = self.metadata.stems
        self.audio_files = self.metadata.audio_files

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        feature_values = []
        for feature in self.features:
            # TODO modularize
            if feature == 'wav':
                audio = ppgs.load.audio(self.metadata.audio_files[index])
                feature_values.append(audio)
            elif feature == 'phonemes':
                try:
                    #prep phoneme mapping in pyfoal
                    # Load alignment
                    alignment = pypar.Alignment(self.cache / f'{stem}.textgrid')

                    hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
                    num_frames = self.metadata.lengths[index]
                    times = np.linspace(hopsize/2, (num_frames-1)*hopsize+hopsize/2, num_frames)
                    times[-1] = alignment.duration()
                    with ppgs_phoneme_list():
                        indices = pyfoal.convert.alignment_to_indices(
                            alignment,
                            hopsize=hopsize,
                            return_word_breaks=False,
                            times=times)
                except ValueError as e:
                    raise ValueError(f'error processing alignment for stem {stem} with error: {e}')
                indices = torch.tensor(indices, dtype=torch.long)
                feature_values.append(indices)
            elif feature == 'stem':
                feature_values.append(stem)
            elif feature == 'audio_file':
                feature_values.append(self.audio_files[index])
            elif feature == 'length': #must immediately follow a feature
                try:
                    feature_values.append(feature_values[-1].shape[-1])
                except AttributeError:
                    feature_values.append(len(feature_values[-1]))
            else:
                try:
                    feature_values.append(torch.load(self.cache / f"{stem}-{feature}.pt"))
                except FileNotFoundError:
                    raise FileNotFoundError(f"Failed to find stem {stem} for feature {feature}")
        return feature_values

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