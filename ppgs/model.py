"""model.py - model definition"""

import functools

import torch


###############################################################################
# Model
###############################################################################


class Model(torch.nn.Module):
    """Model definition"""

    # TODO - add hyperparameters as input args
    def __init__(self):
        super().__init__()

        # TODO - define model
        raise NotImplementedError

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self):
        """Perform model inference"""
        # TODO - define model arguments and implement forward pass
        raise NotImplementedError



class BaselineModel(torch.nn.Sequential):

    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels=128,
        kernel_size=5):
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        super().__init__(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, output_channels))