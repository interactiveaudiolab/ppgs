from encodec import EncodecModel
from types import MethodType
import torch

CONFIG = 'transformerencodec4096'
MODULE = 'ppgs'

INPUT_CHANNELS = 4096 #dimensionality of encodec latents
REPRESENTATION = 'encodec'
MODEL = 'transformer'
NUM_WORKERS=10
EVALUATION_BATCHES = 16
NUM_STEPS = 300000

def _decode(self, q_indices: torch.Tensor) -> torch.Tensor:
    quantized_out = torch.empty((q_indices.shape[1], 4096, q_indices.shape[2]), device=q_indices.device)
    for i, indices in enumerate(q_indices):
        layer = self.layers[i]
        quantized = layer.decode(indices)
        quantized_out[:, i*128:(i+1)*128, :] = quantized
    return quantized_out

def _frontend(device='cpu'):
    import torch
    quantizer = EncodecModel.encodec_model_24khz().quantizer
    quantizer.vq.decode = MethodType(_decode, quantizer.vq)
    quantizer.to(device)

    def _quantize(batch: torch.Tensor):
        batch = batch.to(torch.int)
        batch = batch.transpose(0, 1)
        return quantizer.decode(batch)

    return _quantize

FRONTEND = _frontend
