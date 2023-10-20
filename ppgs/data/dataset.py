import contextlib
import json
from pathlib import Path

import accelerate
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

    def __init__(self, name_or_files, partition=None, features=['audio']):
        self.features = features
        self.metadata = Metadata(name_or_files, partition=partition)
        self.cache = self.metadata.cache
        self.stems = self.metadata.stems
        self.audio_files = self.metadata.audio_files

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        feature_values = []
        if isinstance(self.features, str):
            self.features = [self.features]
        for feature in self.features:

            # Load audio
            if feature == 'audio':
                audio = ppgs.load.audio(self.audio_files[index])
                feature_values.append(audio)

            # Load phoneme alignment
            elif feature == 'phonemes':

                # Load alignment
                alignment = pypar.Alignment(self.cache / f'{stem}.TextGrid')

                # Lowercase and replace silence tokens
                for i in range(len(alignment)):
                    if str(alignment[i]) == '[SIL]':
                        alignment[i].word = pypar.SILENCE
                    for j in range(len(alignment[i])):
                        if str(alignment[i][j]) == '[SIL]':
                            alignment[i][j].phoneme = pypar.SILENCE
                        else:
                            alignment[i][j].phoneme = \
                                str(alignment[i][j]).lower()

                # Convert to indices
                hopsize = ppgs.HOPSIZE / ppgs.SAMPLE_RATE
                num_frames = self.metadata.lengths[index]
                times = np.linspace(
                    hopsize / 2,
                    (num_frames - 1) * hopsize + hopsize / 2,
                    num_frames)
                times[-1] = alignment.duration()
                with ppgs_phoneme_list():
                    indices = pyfoal.convert.alignment_to_indices(
                        alignment,
                        hopsize=hopsize,
                        return_word_breaks=False,
                        times=times)
                indices = torch.tensor(indices, dtype=torch.long)
                feature_values.append(indices)

            # Add stem
            elif feature == 'stem':
                feature_values.append(stem)

            # Add filename
            elif feature == 'audio_file':
                feature_values.append(self.audio_files[index])

            # Add length
            elif feature == 'length':
                try:
                    feature_values.append(feature_values[-1].shape[-1])
                except AttributeError:
                    feature_values.append(len(feature_values[-1]))

            # Add input representation
            else:
                feature_values.append(
                    torch.load(self.cache / f'{stem}-{feature}.pt'))

        return feature_values

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)

    def buckets(self):
        """Partition data into buckets based on length to minimize padding"""
        # Prevent errors when using small datasets
        if len(self) == 0:
            raise ValueError('Dataset has 0 items, cannot bucket')
        num_buckets = max(1, min(ppgs.BUCKETS, len(self)))

        # Get the size of a bucket
        size = len(self) // num_buckets

        # Get indices in order of length
        lengths = self.metadata.lengths
        indices = np.argsort(lengths)

        # Split into buckets based on length
        try:
            buckets = [indices[i:i + size] for i in range(0, len(self), size)]
        except ValueError as error:
            import pdb; pdb.set_trace()
            pass

        # Add max length of each bucket
        return [(lengths[bucket[-1]], bucket) for bucket in buckets]


###############################################################################
# Utilities
###############################################################################


class Metadata:

    def __init__(self, name_or_files, partition=None, overwrite_cache=False):
        """Create a metadata object for the given dataset or sources"""
        with accelerate.state.PartialState().main_process_first():
            lengths = {}

            # Create dataset from string identifier
            if isinstance(name_or_files, str):
                self.name = name_or_files
                self.cache = ppgs.CACHE_DIR / self.name

                # Get stems corresponding to partition
                partition_dict = ppgs.load.partition(self.name)
                if partition is not None:
                    self.stems = partition_dict[partition]
                    lengths_file = self.cache / f'{partition}-lengths.json'
                else:
                    self.stems = sum(partition_dict.values(), start=[])
                    lengths_file = self.cache / f'lengths.json'

                # Get audio filenames
                self.audio_files = [
                    self.cache / (stem + '.wav') for stem in self.stems]

                # Maybe remove previous cached lengths
                if overwrite_cache:
                     lengths_file.unlink(missing_ok=True)

                # Load cached lengths
                if lengths_file.exists():
                    with open(lengths_file, 'r') as f:
                        lengths = json.load(f)

            # Create dataset from a list of audio filenames
            else:
                self.name = '<list of files>'
                self.audio_files = name_or_files
                self.stems = [Path(file).stem for file in self.audio_files]
                self.cache = None

            if not lengths:

                # Compute length in frames
                for stem, audio_file in zip(self.stems, self.audio_files):
                    lengths[stem] = \
                        torchaudio.info(audio_file).num_frames // ppgs.HOPSIZE

                # Maybe cache lengths
                if self.cache is not None:
                    with open(lengths_file, 'w+') as file:
                        json.dump(lengths, file)

            # Match ordering
            self.lengths = [lengths[stem] for stem in self.stems]

    def __len__(self):
        return len(self.stems)


@contextlib.contextmanager
def ppgs_phoneme_list():
    """Context manager for changing the default phoneme mapping of pyfoal"""
    # Get current state
    previous = getattr(pyfoal.convert.phoneme_to_index, 'map', None)

    try:

        # Change state
        setattr(
            pyfoal.convert.phoneme_to_index,
            'map',
            ppgs.PHONEME_TO_INDEX_MAPPING)

        # Execute user code
        yield

    finally:

        # Restore state
        if previous is not None:
            setattr(pyfoal.convert.phoneme_to_index, 'map', previous)
        else:
            delattr(pyfoal.convert.phoneme_to_index, 'map')
