import torch

import ppgs


###############################################################################
# Custom samplers
###############################################################################


class Sampler(torch.utils.data.sampler.BatchSampler):

    def __init__(self, dataset):
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
        for max_length, bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Get current batch size
            size = ppgs.MAX_FRAMES // max_length

            # Make batches
            batches.extend(
                [bucket[i:i + size] for i in range(0, len(bucket), size)])

        # Shuffle batches
        return [
            batches[i] for i in
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch
