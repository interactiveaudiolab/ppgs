import torch

import ppgs


def loaders(dataset, representation='senone'):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train', representation), loader(dataset, 'valid', representation)


def loader(dataset, partition, representation='senone'):
    """Retrieve a data loader"""
    dataset_object = ppgs.data.Dataset(dataset, partition, representation)
    return torch.utils.data.DataLoader(
        dataset=dataset_object,
        batch_size=1 if partition == 'test' else ppgs.BATCH_SIZE,
        sampler=ppgs.data.sampler(dataset_object, partition),
        num_workers=ppgs.NUM_WORKERS,
        pin_memory=True,
        collate_fn=ppgs.data.collate)
