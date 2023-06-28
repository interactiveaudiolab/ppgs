import torch

import ppgs


def loaders(dataset, representation=ppgs.REPRESENTATION, reduced_features=False):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train', representation, reduced_features), loader(dataset, 'valid', representation, reduced_features)


def loader(dataset, partition, representation=ppgs.REPRESENTATION, reduced_features=False):
    """Retrieve a data loader"""
    dataset_object = ppgs.data.Dataset(dataset, partition, representation, reduced_features)
    if partition == 'test':
        return torch.utils.data.DataLoader(
            dataset=dataset_object,
            num_workers=ppgs.NUM_WORKERS,
            pin_memory=True,
            collate_fn=ppgs.data.collate
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset_object,
            # batch_size=1 if partition == 'test' else ppgs.BATCH_SIZE,
            # sampler=ppgs.data.sampler(dataset_object, partition),
            batch_sampler=ppgs.data.sampler(dataset_object, partition),
            num_workers=ppgs.NUM_WORKERS,
            pin_memory=True,
            collate_fn=ppgs.data.collate if not reduced_features else ppgs.data.reduced_collate)
