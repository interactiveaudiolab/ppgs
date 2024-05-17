import json
from pathlib import Path

import huggingface_hub
import torch
import torchutil
import torchaudio

import ppgs


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    path = Path(file)
    if path.suffix.lower() == '.mp3':
        try:
            audio, sample_rate = torchaudio.load(path, format='mp3')
        except RuntimeError:
            raise RuntimeError(
                'Failed to load mp3 file, make sure ffmpeg<=4.3 is installed')
    else:
        audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return ppgs.resample(audio, sample_rate)


def model(checkpoint=None, representation=None):
    """Load a model"""
    if representation is not None:
        if representation == 'w2v2fb':
            checkpoint = huggingface_hub.hf_hub_download(
                'CameronChurchwell/ppgs',
                'w2v2fb-425k.pt')
            conf = vars(ppgs.config.w2v2fb)
            conf = {k: v for k, v in conf.items() if not k.startswith('__')}
            kwargs = {kv[0].lower() : kv[1] for kv in conf.items()}
        elif representation == 'mel':
            kwargs = {}
        else:
            raise ValueError(
                'Supplying representation directly only supported '
                'for w2v2fb and mel')
    else:
        kwargs = {}

    model = ppgs.Model(**kwargs)

    # Pretrained model
    if ppgs.MODEL in ['W2V2FC', 'W2V2FS']:
        return model

    # Maybe download from HuggingFace
    if checkpoint is None and ppgs.LOCAL_CHECKPOINT is None:
        if ppgs.REPRESENTATION == 'mel' or ppgs.REPRESENTATION is None:
            checkpoint = huggingface_hub.hf_hub_download(
                'CameronChurchwell/ppgs',
                'mel-800k.pt')
        elif ppgs.REPRESENTATION == 'w2v2fb':
            checkpoint = huggingface_hub.hf_hub_download(
                'CameronChurchwell/ppgs',
                'w2v2fb-425k.pt')
        else:
            raise ValueError(
                f'No default checkpoints exist for '
                f'representation {ppgs.REPRESENTATION}')
    elif checkpoint is None and ppgs.LOCAL_CHECKPOINT is not None:
        checkpoint = ppgs.LOCAL_CHECKPOINT

    # Load from checkpoint
    state_dict = torch.load(checkpoint, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)

    return model


def partition(dataset):
    """Load partitions for dataset"""
    with open(ppgs.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def phoneme_weights(device='cpu'):
    """Load phoneme weights for class balancing"""
    if not hasattr(phoneme_weights, 'weights'):
        try:

            # Load and cache weights
            phoneme_weights.weights = torch.load(ppgs.CLASS_WEIGHT_FILE)

        except FileNotFoundError:

            # Setup dataloader
            loader = ppgs.data.loader(
                ppgs.TRAINING_DATASET,
                partition='train',
                features=['phonemes', 'length'])

            # Get phoneme counts
            counts = torch.zeros(40, dtype=torch.long).to(device)
            for phonemes, lengths in torchutil.iterator(
                loader,
                'Computing phoneme frequencies for class balancing',
                total=len(loader)
            ):
                phonemes, lengths = phonemes.to(device), lengths.to(device)
                mask = ppgs.model.transformer.mask_from_lengths(lengths)
                phonemes = phonemes[mask]
                counts.scatter_add_(
                    dim=0,
                    index=phonemes,
                    src=torch.ones_like(phonemes))

            # Compute weights from counts
            phoneme_weights.weights = counts.min() / counts

            # Save
            torch.save(phoneme_weights.weights, ppgs.CLASS_WEIGHT_FILE)

    return phoneme_weights.weights.to(device)
