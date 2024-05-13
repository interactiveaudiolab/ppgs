import json
import warnings
from pathlib import Path

# import accelerate
import numpy as np
import pypar
import torch
import torchaudio

import ppgs


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name_or_files,
        partition=None,
        features=['audio'],
        max_frames=ppgs.MAX_TRAINING_FRAMES):
        self.features = features
        self.metadata = Metadata(
            name_or_files,
            partition=partition,
            max_frames=max_frames)
        self.cache = self.metadata.cache
        self.stems = self.metadata.stems
        self.audio_files = self.metadata.audio_files
        self.lengths = self.metadata.lengths

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
                indices = alignment.framewise_phoneme_indices(
                    ppgs.PHONEME_TO_INDEX_MAPPING,
                    hopsize,
                    times)
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
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // ppgs.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)
        lengths = np.sort(self.lengths)

        # Split into buckets based on length
        buckets = [
            np.stack((indices[i:i + size], lengths[i:i + size])).T
            for i in range(0, len(self), size)]

        # Concatenate partial bucket
        if len(buckets) == ppgs.BUCKETS + 1:
            residual = buckets.pop()
            buckets[-1] = np.concatenate((buckets[-1], residual), axis=0)

        return buckets


###############################################################################
# Utilities
###############################################################################


class Metadata:

    def __init__(
        self,
        name_or_files,
        partition=None,
        overwrite_cache=False,
        max_frames=ppgs.MAX_TRAINING_FRAMES):
        """Create a metadata object for the given dataset or sources"""
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
            self.stems = [
                Path(file).parent / Path(file).stem
                for file in self.audio_files]
            self.cache = None

        if not lengths:

            # Compute length in frames
            for stem, audio_file in zip(self.stems, self.audio_files):
                info = torchaudio.info(audio_file)
                length = int(
                    info.num_frames * (ppgs.SAMPLE_RATE / info.sample_rate)
                ) // ppgs.HOPSIZE

                # Omit if length is too long to avoid OOM
                if length <= max_frames:
                    lengths[stem] = length
                else:
                    warnings.warn(
                        f'File {audio_file} of length {length} '
                        f'exceeds max_frames of {max_frames}. Skipping.')

            # Maybe cache lengths
            if self.cache is not None:
                with open(lengths_file, 'w+') as file:
                    json.dump(lengths, file)

        # Match ordering
        (
            self.audio_files,
            self.stems,
            self.lengths
            ) = zip(*[
            (file, stem, lengths[stem])
            for file, stem in zip(self.audio_files, self.stems)
            if stem in lengths
        ])

    def __len__(self):
        return len(self.stems)
