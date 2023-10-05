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


def model(checkpoint=ppgs.DEFAULT_CHECKPOINT):
    """Load a model"""
    model = ppgs.Model()

    # Pretrained model
    if ppgs.MODEL in ['W2V2FC', 'W2V2FS']:
        return model

    # Load from checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

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
            for phonemes, lengths in ppgs.iterator(
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
