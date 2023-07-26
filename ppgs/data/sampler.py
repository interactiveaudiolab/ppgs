"""sampler.py - data sampling"""

import torch
import math
import ppgs

###############################################################################
# Batch sampler
###############################################################################


def sampler(dataset, partition):
    """Create batch index sampler"""
    # Get sampler indices
    indices = list(range(len(dataset)))

    # Maybe use distributed sampler for training
    if partition == 'train':
        print("creating train sampler")
        # return Sampler(indices)
        return Sampler(dataset)

    # Possibly deterministic random sampler for validation
    elif partition == 'valid':
        print("creating valid sampler")
        # return Sampler(indices)
        return Sampler(dataset)

    # Sample test data sequentially
    elif partition == 'test':
        print("creating test sampler")
        return torch.utils.data.SequentialSampler(indices)

    else:
        raise ValueError(f'Partition {partition} is not implemented')


###############################################################################
# Custom samplers
###############################################################################

class Sampler:

    def __init__(self, dataset):
        self.epoch = 0
        self.length = len(dataset)
        self.buckets = dataset.buckets()

    def __iter__(self):
        return iter(self.batch())

    def __len__(self):
        return len(self.batch())

    def batch(self):
        """Produces batch indices for one epoch"""
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(ppgs.RANDOM_SEED + self.epoch)

        # Make variable-length batches with roughly equal number of frames
        batches = []
        for max_length, bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Get current batch size
            size = ppgs.MAX_FRAMES // max_length

            # Make batches
            batches.extend(
                [bucket[i:i + size] for i in range(0, len(bucket), size)])

        # Shuffle
        return [
            batches[i] for i in 
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedSampler:

    def __init__(self, indices):
        self.indices = indices
        self.epoch = 0
        self.rank = torch.distributed.get_rank()
        self.num_replicas = torch.distributed.get_world_size()
        self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(ppgs.RANDOM_SEED + self.epoch)
        indices = [
            self.indices[i] for i in
            torch.randperm(len(self.indices), generator=generator)]

        # Add extra samples to make it evenly divisible
        padding = self.total_size - len(indices)
        if padding <= len(indices):
            indices += indices[:padding]
        else:
            indices += (
                indices * math.ceil(padding / len(indices)))[:padding]

        # Subsample
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch