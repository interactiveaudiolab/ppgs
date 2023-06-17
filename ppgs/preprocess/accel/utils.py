import torch
import torchaudio
import ppgs
import numpy as np
import warnings
from pathlib import Path

MAX_DURATION = 320 #max total duration for a batch in seconds
MAX_SAMPLES = MAX_DURATION * ppgs.SAMPLE_RATE

class Metadata:

    def __init__(self, dataset_or_files, partition=None):
        if isinstance(dataset_or_files, str):
            self.name = dataset_or_files
            self.cache = ppgs.CACHE_DIR / self.name
            # if partition is not None:
            #     self.stems = [self.cache / stem for stem in ppgs.load.partition(name)[partition]]
            # else:
            #     self.stems = [self.cache / audio_file.stem for audio_file in self.cache.glob('*.wav')]
            if partition is not None:
                self.audio_files = [self.cache / (stem + '.wav') for stem in ppgs.load.partition(self.name)[partition]]
            else:
                #TODO include non-cached version
                self.audio_files = list(self.cache.glob('*.wav'))
        elif isinstance(dataset_or_files, list):
            self.name = "list of files"
            self.cache = None
            self.audio_files = dataset_or_files
        elif isinstance(dataset_or_files, Path):
            self.name = dataset_or_files.stem
            self.cache = None
            self.audio_files = [dataset_or_files]
        else:
            import pdb; pdb.set_trace()
            raise ValueError("need to pass either a name of a dataset or a list of files")

        print('preparing dataset metadata (operation may be slow)')
        # Store lengths for bucketing
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.lengths = [torchaudio.info(audio_file).num_frames for audio_file in self.audio_files]

    def __len__(self):
        return len(self.stems)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_or_files, partition=None):
        self.metadata = Metadata(dataset_or_files, partition)
        self.audio_files = self.metadata.audio_files

    def __getitem__(self, index):
        audio_file = self.metadata.audio_files[index]

        audio = ppgs.load.audio(audio_file)

        return audio, audio_file, audio.shape[-1]

    def __len__(self):
        return len(self.audio_files)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // ppgs.BUCKETS

        # Get indices in order of length
        lengths = []
        lengths = self.metadata.lengths
        indices = np.argsort(lengths)

        # Split into buckets based on length
        buckets = [indices[i:i + size] for i in range(0, len(self), size)]

        # Add max length of each bucket
        buckets = [(lengths[bucket[-1]], bucket) for bucket in buckets]

        return buckets

class Sampler:

    def __init__(self, dataset):
        print('creating preprocessing sampler')
        buckets = dataset.buckets()
        # Make variable-length batches with roughly equal number of frames
        batches = []
        for max_length, bucket in reversed(buckets):

            # Get current batch size
            # size = min(128, ppgs.MAX_FRAMES // max_length)
            size = min(512, MAX_SAMPLES // max_length)

            # Make batches
            batches.extend(
                [bucket[i:i + size] for i in range(0, len(bucket), size)])

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def collate(batch):
    audio, audio_files, lengths = zip(*batch)
    max_length = max(lengths)
    batch_size = len(audio_files)
    padded_audio = torch.zeros(
        (batch_size, 1, max_length),
        dtype=torch.float)
    for i, a in enumerate(audio):
        padded_audio[i, 0, :a.shape[-1]] = a[0]
    lengths = torch.tensor(lengths)
    return padded_audio, audio_files, lengths

def loader(dataset_or_files, partition=None, num_workers=0):
    dataset_object = Dataset(dataset_or_files, partition)
    loader_object = torch.utils.data.DataLoader(
        dataset=dataset_object,
        pin_memory=True,
        num_workers=num_workers,
        batch_sampler=Sampler(dataset_object),
        collate_fn=collate
    )
    return loader_object

def save_masked(tensor, file, length):
    try:
        sub_tensor = tensor[:, :length].clone()
        torch.save(sub_tensor, file)
    except Exception as e:
        print(f'error saving file {file}: {e}', flush=True)