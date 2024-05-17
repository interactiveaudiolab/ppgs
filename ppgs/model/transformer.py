import math

import torch

import ppgs


###############################################################################
# Transformer model
###############################################################################


class Transformer(torch.nn.Module):

    def __init__(
        self,
        num_hidden_layers=ppgs.NUM_HIDDEN_LAYERS,
        hidden_channels=ppgs.HIDDEN_CHANNELS,
        input_channels=ppgs.INPUT_CHANNELS,
        output_channels=ppgs.OUTPUT_CHANNELS,
        kernel_size=ppgs.KERNEL_SIZE,
        attention_heads=ppgs.ATTENTION_HEADS,
        is_causal=ppgs.IS_CAUSAL
    ):
        super().__init__()
        self.position = PositionalEncoding(hidden_channels)
        self.input_layer = torch.nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding='same')
        self.model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_channels, attention_heads),
            num_hidden_layers)
        self.output_layer = torch.nn.Conv1d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            padding='same')
        self.is_causal = is_causal

    def forward(self, x, lengths):
        if self.is_causal:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                torch.max(lengths),
                device = x.device
            )
        else:
            causal_mask = None
        mask = mask_from_lengths(lengths).unsqueeze(1)
        x = self.input_layer(x) * mask
        x = self.model(
            self.position(x.permute(2, 0, 1)),
            mask=causal_mask,
            src_key_padding_mask=~mask.squeeze(1)
        ).permute(1, 2, 0)
        return self.output_layer(x) * mask


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


def mask_from_lengths(lengths, padding=0):
    """Create boolean mask from sequence lengths and offset to start"""
    x = torch.arange(
        lengths.max() + 2 * padding,
        dtype=lengths.dtype,
        device=lengths.device)
    return x.unsqueeze(0) - 2 * padding < lengths.unsqueeze(1)
