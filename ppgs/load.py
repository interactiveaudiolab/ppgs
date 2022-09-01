import json
import torch
import torchaudio

import ppgs


###############################################################################
# Loading utilities
###############################################################################

def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return ppgs.resample(audio, sample_rate)

def partition(dataset):
    """Load partitions for dataset"""
    with open(ppgs.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
