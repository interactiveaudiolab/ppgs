"""model.py - model definition"""

import functools
import ppgs
import torch


###############################################################################
# Model
###############################################################################


class BaselineModel(torch.nn.Sequential):
    """Create a baseline model to compare with. Basedon on torch Sequential superclass."""

    def __init__(
        self,
        input_channels=None, #dimensionality of input time series
        output_channels=None, #phoneme time series dimensionality
        hidden_channels=128,
        kernel_size=5):

        if input_channels is None:
            input_channels = ppgs.INPUT_CHANNELS
        if output_channels is None:
            output_channels = ppgs.OUTPUT_CHANNELS

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
