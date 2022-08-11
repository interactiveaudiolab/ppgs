"""dataset.py - data loading"""


import torch

import ppgs


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset

    Arguments
        name - string
            The name of the dataset
        partition - string
            The name of the data partition
    """

    def __init__(self, name, partition):
        # TODO - initialize dataset
        self.stems = None

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        stem = self.stems[index]

        # TODO - Load from stem
        raise NotImplementedError

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)
