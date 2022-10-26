"""model.py - model definition"""

import functools
import ppgs
import torch


###############################################################################
# Model
###############################################################################


class BaselineModel(torch.nn.Sequential):

    def __init__(
        self,
        input_channels=ppgs.INPUT_CHANNELS, #dimensionality of input time series
        output_channels=ppgs.OUTPUT_CHANNELS, #phoneme time series dimensionality
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
