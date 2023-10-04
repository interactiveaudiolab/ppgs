import functools

import torch

import ppgs


###############################################################################
# Convolution model
###############################################################################


class Convolution(torch.nn.Sequential):
    """Simple convolutional model"""

    def __init__(self):
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=ppgs.KERNEL_SIZE,
            padding='same')
        super().__init__(
            conv_fn(ppgs.INPUT_CHANNELS, ppgs.HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            conv_fn(ppgs.HIDDEN_CHANNELS, ppgs.HIDDEN_CHANNELS),
            torch.nn.ReLU(),
            conv_fn(ppgs.HIDDEN_CHANNELS, len(ppgs.PHONEMES)))

    def forward(self, x, _):
        return super().forward(x)
