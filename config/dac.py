import dac
import torch

MODULE = 'ppgs'

# Configuration name
CONFIG = 'dac'

# Dimensionality of input representation
INPUT_CHANNELS = 96

# Input representation
REPRESENTATION = 'dac'

# Number of training steps
STEPS = 200000

def _frontend(device='cpu'):
    model_path = dac.utils.download(model_type='16khz')
    model = dac.DAC.load(model_path)
    model = model.to(device)

    def _quantize(batch: torch.Tensor):
        batch = batch.to(torch.int)
        z, latents, codes = model.quantizer.from_codes(batch)
        return latents


    return _quantize

# This function takes as input a torch.Device and returns a callable frontend
FRONTEND = _frontend
