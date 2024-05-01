import torch

import mamba_ssm

import ppgs

from functools import partial

class Mamba(torch.nn.Module):

    def __init__(
        self,
        num_layers=ppgs.NUM_HIDDEN_LAYERS,
        channels=ppgs.HIDDEN_CHANNELS):
        super().__init__()
        self.input_layer = torch.nn.Conv1d(
            ppgs.INPUT_CHANNELS,
            ppgs.HIDDEN_CHANNELS,
            kernel_size=ppgs.KERNEL_SIZE,
            padding='same')
        self.layers = torch.nn.ModuleList([
            # mamba_ssm.Mamba(d_model=ppgs.HIDDEN_CHANNELS)
            mamba_ssm.modules.mamba_simple.Block(
                mixer_cls=partial(mamba_ssm.Mamba, layer_idx=i),
                norm_cls=torch.nn.LayerNorm,
                dim=ppgs.HIDDEN_CHANNELS
            )
            for i in range(ppgs.NUM_HIDDEN_LAYERS)
        ])
        self.output_layer = torch.nn.Conv1d(
            ppgs.HIDDEN_CHANNELS,
            ppgs.OUTPUT_CHANNELS,
            kernel_size=ppgs.KERNEL_SIZE,
            padding='same')


    def forward(self, x, lengths):
        mask = mask_from_lengths(lengths).unsqueeze(1)
        x = self.input_layer(x) * mask
        hidden_states = x.permute(0, 2, 1)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        x = hidden_states.permute(0, 2, 1)
        return self.output_layer(x) * mask


###############################################################################
# Utilities
###############################################################################

def mask_from_lengths(lengths, padding=0):
    """Create boolean mask from sequence lengths and offset to start"""
    x = torch.arange(
        lengths.max() + 2 * padding,
        dtype=lengths.dtype,
        device=lengths.device)
    return x.unsqueeze(0) - 2 * padding < lengths.unsqueeze(1)