from encodec import EncodecModel

MODULE = 'ppgs'

# Configuration name
CONFIG = 'encodec-7-layers'

# Dimensionality of input representation
INPUT_CHANNELS = 128

NUM_HIDDEN_LAYERS = 7

# Input representation
REPRESENTATION = 'encodec'

def _frontend(device='cpu'):
    import torch
    quantizer = EncodecModel.encodec_model_24khz().quantizer
    quantizer.to(device)

    def _quantize(batch: torch.Tensor):
        batch = batch.to(torch.int)
        batch = batch.transpose(0, 1)
        return quantizer.decode(batch)

    return _quantize

# This function takes as input a torch.Device and returns a callable frontend
FRONTEND = _frontend

CLIPPING_QUANTILE = 0.8