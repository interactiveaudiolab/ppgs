import math

import torch

import ppgs


def mask_from_lengths(lengths, padding=0):
    """Create boolean mask from sequence lengths and offset to start. shape is batch x 1 x max_length"""
    x = torch.arange(lengths.max()+2*padding, dtype=lengths.dtype, device=lengths.device)

    return (x.unsqueeze(0)-2*padding < lengths.unsqueeze(1))

###############################################################################
# Transformer stack
###############################################################################


class Transformer(torch.nn.Module):

    def __init__(self, num_layers=ppgs.NUM_HIDDEN_LAYERS, channels=ppgs.HIDDEN_CHANNELS):
        super().__init__()
        self.position = PositionalEncoding(channels, .1)
        self.model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(channels, ppgs.ATTENTION_HEADS),
            num_layers)

    def forward(self, x, lengths):
        mask = mask_from_lengths(lengths)
        return self.model(
            self.position(x.permute(2, 0, 1)),
            src_key_padding_mask=~mask.squeeze(1)
        ).permute(1, 2, 0)


###############################################################################
# Utilities
###############################################################################


class PositionalEncoding(torch.nn.Module):

    def __init__(self, channels, dropout=.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        index = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(max_len, 1, channels)
        encoding[:, 0, 0::2] = torch.sin(index * frequency)
        encoding[:, 0, 1::2] = torch.cos(index * frequency)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        if x.size(0) > self.encoding.size(0):
            raise ValueError('size is too large')
        return self.dropout(x + self.encoding[:x.size(0)])