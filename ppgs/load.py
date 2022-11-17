import json
import torch
import torchaudio

import ppgs
from collections import OrderedDict

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

def ddp_to_single_state_dict(state_dict):
    """Convert a DDP model state dict to one which can be loaded on a single device"""

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
