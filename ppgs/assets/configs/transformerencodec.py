from encodec import EncodecModel

CONFIG = 'transformerencodec'
MODULE = 'ppgs'

INPUT_CHANNELS = 128 #dimensionality of encodec latents
REPRESENTATION = 'encodec'
MODEL = 'transformer'
NUM_WORKERS=10
EVALUATION_BATCHES = 16

BATCH_SIZE = 512


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
