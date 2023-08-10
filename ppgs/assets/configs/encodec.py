from encodec import EncodecModel
from types import MethodType
import torch

CONFIG = 'encodec'
MODULE = 'ppgs'

INPUT_CHANNELS = 128 #dimensionality of encodec latents
REPRESENTATION = 'encodec'
MODEL = 'transformer'

EVALUATION_BATCHES = 16

NUM_HIDDEN_LAYERS = 5
MAX_FRAMES = 100000
HIDDEN_CHANNELS = 512

def _frontend(device='cpu'):
    import torch
    quantizer = EncodecModel.encodec_model_24khz().quantizer
    quantizer.to(device)

    def _quantize(batch: torch.Tensor):
        batch = batch.to(torch.int)
        batch = batch.transpose(0, 1)
        return quantizer.decode(batch)

    return _quantize

FRONTEND = _frontend
