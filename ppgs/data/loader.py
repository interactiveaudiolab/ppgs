import torch

import ppgs


def loaders(dataset, features=[ppgs.REPRESENTATION, 'phonemes', 'length', 'stem']):
    """Retrieve data loaders for training and evaluation"""
    return loader(dataset, 'train', features), loader(dataset, 'valid', features)


def loader(dataset, partition, features=[ppgs.REPRESENTATION, 'phonemes', 'length', 'stem']):
    """Retrieve a data loader"""
    dataset_object = ppgs.data.Dataset(dataset, partition, features)
    collator_object = ppgs.data.Collator(features)
    if partition == 'test':
        return torch.utils.data.DataLoader(
            dataset=dataset_object,
            num_workers=ppgs.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collator_object,
            batch_size=128
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset_object,
            batch_sampler=ppgs.data.sampler(dataset_object, partition),
            num_workers=ppgs.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collator_object)
