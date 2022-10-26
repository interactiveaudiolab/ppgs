import torch

import ppgs


def loaders(dataset, representation='ppg'):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train', representation), loader(dataset, 'valid', representation)


def loader(dataset, partition, representation='ppg'):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=ppgs.data.Dataset(dataset, partition, representation),
        batch_size=1 if partition == 'test' else ppgs.BATCH_SIZE,
        shuffle=partition == 'train',
        num_workers=ppgs.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ppgs.data.collate)
