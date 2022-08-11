import torch

import ppgs


def loaders(dataset):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train'), loader(dataset, 'valid')


def loader(dataset, partition):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=ppgs.data.Dataset(dataset, partition),
        batch_size=ppgs.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=ppgs.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ppgs.data.collate)
