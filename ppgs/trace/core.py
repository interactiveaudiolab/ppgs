import contextlib
import functools
import itertools
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import torchutil

import ppgs

FIXED_LENGTH_TIME = 5 # seconds
FIXED_LENGTH_SIZE = ppgs.SAMPLE_RATE * FIXED_LENGTH_TIME # samples

def trace(
    input_file: os.PathLike,
    module_file: os.PathLike,
    representation: Optional[str] = None,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        input_file
            The audio file
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            Phonetic posteriorgram
            shape=(len(ppgs.PHONEMES), frames)
    """
    module = from_file(
        file=input_file,
        representation=representation,
        checkpoint=checkpoint,
        gpu=gpu
    )
    module.save(str(module_file))

def from_file(
    file: os.PathLike,
    representation: Optional[str] = None,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        file
            The audio file
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            Phonetic posteriorgram
            shape=(len(ppgs.PHONEMES), frames)
    """
    # Load audio
    audio = ppgs.load.audio(file)
    repeats = (FIXED_LENGTH_SIZE // audio.shape[-1]) + 1
    audio = audio.repeat(1, repeats).clone()[..., :FIXED_LENGTH_SIZE]
    print("Using audio shape: ", audio.shape)

    # Compute PPGs
    return from_audio(
        audio=audio,
        sample_rate=ppgs.SAMPLE_RATE,
        representation=representation,
        checkpoint=checkpoint,
        gpu=gpu
    )

def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    representation: Optional[str] = None,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: int = None
) -> torch.Tensor:
    """Infer ppgs from audio

    Arguments
        audio
            Batched audio to process
            shape=(batch, 1, samples)
        sample_rate
            Audio sampling rate
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            Phonetic posteriorgrams
            shape=(batch, len(ppgs.PHONEMES), frames)
    """
    # Preprocess
    if representation is None:
        representation = ppgs.REPRESENTATION

    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    audio = audio.to(device)

    features_from_audio = getattr(ppgs.preprocess, representation).from_audio

    preprocess_fn = functools.partial(features_from_audio,
        sample_rate=sample_rate,
        gpu=gpu
    )
    module = traced_inference(
        audio=audio,
        preprocess_fn=preprocess_fn,
        representation=representation,
        checkpoint=checkpoint,
    )

    return module

class StreamlineWrapper(torch.nn.Module):
    def __init__(self, preprocess_fn, model):
        super().__init__()
        self.preprocess_fn = preprocess_fn
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, audio):
        assert audio.shape[0] == 1, "tracing only works on unbatched inputs because of lengths here"
        features = self.preprocess_fn(audio).to(torch.float32)
        lengths = torch.tensor([features.shape[-1]], device=features.device)
        logits = self.model(features, lengths)
        ppg = self.softmax(logits)
        return ppg

def traced_inference(audio, preprocess_fn, representation='mel', checkpoint=None):
    """Perform model inference"""

    # Skip inference if we want input representations
    if ppgs.REPRESENTATION_KIND == 'latents':
        return features

    # Maybe cache model
    if not hasattr(traced_inference, 'models'): traced_inference.models = {}
    if (
        not hasattr(traced_inference, 'model') or
        traced_inference.checkpoint != checkpoint or
        traced_inference.device_type != features.device.type or
        traced_inference.representation != representation
    ):
        model_key = str(representation) + str(checkpoint)
        if model_key not in traced_inference.models:
            traced_inference.models[model_key] = ppgs.load.model(checkpoint=checkpoint, representation=representation)
        traced_inference.model = traced_inference.models[model_key]
        traced_inference.checkpoint = checkpoint
        traced_inference.representation = representation
        traced_inference.device_type = audio.device.type

    # Move model to correct device (no-op if devices are the same)
    traced_inference.model = traced_inference.model.to(audio.device)
    traced_inference.model.eval()

    # Infer
    # with torchutil.inference.context(traced_inference.model):
    traced_inference.model = StreamlineWrapper(preprocess_fn, traced_inference.model).to(audio.device)
    # with torch.autocast(audio.device.type, dtype=torch.float16, cache_enabled=False), torch.no_grad():
    with torch.no_grad():
        traced_inference.model(audio)
        module = torch.jit.trace(traced_inference.model, audio)
        return module