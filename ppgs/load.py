import json
from pathlib import Path

import torch
import torchaudio

import ppgs


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    path = Path(file)
    if path.suffix.lower() == '.mp3':
        audio, sample_rate = torchaudio.load(path, format='mp3')
    else:
        audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return ppgs.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(ppgs.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def model(checkpoint=ppgs.DEFAULT_CHECKPOINT):
    """Load a model"""
    model = ppgs.Model()

    # Pretrained model
    if ppgs.MODEL in ['W2V2FC', 'W2V2FS']:
        return model

    # Load from checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    return model
