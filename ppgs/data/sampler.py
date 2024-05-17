import torch

import ppgs


###############################################################################
# Sampler selection
###############################################################################


def sampler(dataset, partition):
    """Create batch sampler"""
    # Deterministic random sampler for training and validation
    if partition.startswith('train') or partition.startswith('valid'):
        return Sampler(dataset)

    # Sample test data sequentially
    elif partition.startswith('test'):
        return torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            1,
            False)

    else:
        raise ValueError(f'Partition {partition} is not defined')


###############################################################################
# Custom samplers
###############################################################################


class Sampler(torch.utils.data.sampler.BatchSampler):

    def __init__(self, dataset, max_frames=ppgs.MAX_TRAINING_FRAMES):
        self.max_frames = max_frames
        self.epoch = 0
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
        for bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Variable batch size
            batch = []
            max_length = 0
            for index, length in bucket:
                max_length = max(max_length, length)
                if (
                    batch and
                    (len(batch) + 1) * max_length > self.max_frames
                ):
                    batches.append(batch)
                    max_length = length
                    batch = [index]
                else:
                    batch.append(index)

            # Don't drop last batch
            if batch:
                batches.append(batch)

        # Shuffle
        return [
            batches[i] for i in
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch
