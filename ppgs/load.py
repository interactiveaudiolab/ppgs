import json
import torch
import torchaudio

import ppgs
from collections import OrderedDict
from pathlib import Path

###############################################################################
# Loading utilities
###############################################################################

def audio(file):
    """Load audio from disk"""
    path = Path(file)
    if path.suffix.lower() == '.mp3':
        # import pdb; pdb.set_trace()
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
    """Load a model from a checkpoint file. Make sure the current configuration values match"""
    try:
        state_dict = torch.load(checkpoint, map_location='cpu')
    except FileNotFoundError:
        raise FileNotFoundError(f'could not find model checkpoint {checkpoint}')
    
    # disregard optimizer from training
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    model = ppgs.Model()()

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        try:
            model.load_state_dict(ddp_to_single_state_dict(state_dict))
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
    
    return model

def ddp_to_single_state_dict(state_dict):
    """Convert a DDP model state dict to one which can be loaded on a single device"""

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
