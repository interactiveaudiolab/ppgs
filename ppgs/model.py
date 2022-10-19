"""model.py - model definition"""

import functools

import torch


###############################################################################
# Model
###############################################################################


class BaselineModel(torch.nn.Sequential):

    def __init__(
        self,
        input_channels=144, #TODO have this filled in dynamically
        output_channels=42, #phonemes
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
