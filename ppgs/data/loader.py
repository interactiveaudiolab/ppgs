import torch

import ppgs


###############################################################################
# Constants
###############################################################################


# Features loaded for training
TRAINING_FEATURES = [ppgs.REPRESENTATION, 'phonemes', 'length']


###############################################################################
# Dataloader
###############################################################################


def loader(
    dataset_or_files,
    partition=None,
    features=TRAINING_FEATURES,
    num_workers=ppgs.NUM_WORKERS,
    max_frames=ppgs.MAX_TRAINING_FRAMES):
    """Retrieve a data loader"""
    # Initialize dataset
    dataset = ppgs.data.Dataset(
        dataset_or_files,
        partition,
        features,
        max_frames)

    # Initialize sampler
    sampler = ppgs.data.Sampler(dataset, max_frames)

    # Initialize dataloader
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ppgs.data.Collate(features))
