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


###############################################################################
# Application programming interface
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: int = None,
    legacy_mode: bool = False
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
        legacy_mode
            Use legacy (unchunked) inference

    Returns
        ppgs
            Phonetic posteriorgrams
            shape=(batch, len(ppgs.PHONEMES), frames)
    """
    # Preprocess
    features = ppgs.preprocess.from_audio(
        audio=audio,
        sample_rate=sample_rate,
        representation=representation,
        gpu=gpu)

    # Get length in frames
    length = torch.tensor([features.shape[-1]], dtype=torch.long)

    # Infer
    return from_features(
        features=features,
        lengths=length,
        representation=representation,
        checkpoint=checkpoint,
        gpu=gpu,
        legacy_mode=legacy_mode)


def from_features(
    features: torch.Tensor,
    lengths: torch.Tensor,
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None,
    softmax: bool = True,
    legacy_mode: bool = False
) -> torch.Tensor:
    """Infer ppgs from input features (e.g. w2v2fb, mel, etc.)

    Arguments
        features
            Input representation
            shape=(batch, len(ppgs.PHONEMES), frames)
        lengths
            The lengths of the features
            shape=(batch,)
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference
        softmax
            Whether to apply softmax normalization to the inferred logits
        legacy_mode
            Use legacy (unchunked) inference

    Returns
        ppgs
            Phonetic posteriorgrams
            shape=(batch, len(ppgs.PHONEMES), frames)
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    if representation is None: # neither mel nor w2v2fb have a frontend

        # Maybe load and cache codebook
        if (
            not hasattr(from_features, 'frontend') and
            ppgs.FRONTEND is not None
        ):
            from_features.frontend = ppgs.FRONTEND(device)

        # Codebook lookup
        if hasattr(from_features, 'frontend'):
            features = from_features.frontend(features.to(device))

    # Infer
    return infer(
        features=features.to(device),
        lengths=lengths.to(device),
        representation=representation,
        checkpoint=checkpoint,
        softmax=softmax,
        legacy_mode=legacy_mode)


def from_file(
    file: Union[str, bytes, os.PathLike],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None,
    legacy_mode: bool = False
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
        legacy_mode
            Use legacy (unchunked) inference

    Returns
        ppgs
            Phonetic posteriorgram
            shape=(len(ppgs.PHONEMES), frames)
    """
    # Load audio
    audio = ppgs.load.audio(file)

    # Compute PPGs
    return from_audio(
        audio=audio,
        sample_rate=ppgs.SAMPLE_RATE,
        representation=representation,
        checkpoint=checkpoint,
        gpu=gpu,
        legacy_mode=legacy_mode
    ).squeeze(0)


def from_file_to_file(
    audio_file: Union[str, bytes, os.PathLike],
    output_file: Union[str, bytes, os.PathLike],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None,
    legacy_mode: bool = False
) -> None:
    """Infer ppg from an audio file and save to a torch tensor file

    Arguments
        audio_file
            The audio file
        output_file
            The .pt file to save PPGs
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference
        legacy_mode
            Use legacy (unchunked) inference
    """
    # Compute PPGs
    result = from_file(
        file=audio_file,
        checkpoint=checkpoint,
        representation=representation,
        gpu=gpu,
        legacy_mode=legacy_mode)

    # Save to disk
    torch.save(result.detach().cpu(), output_file)


def from_files_to_files(
    audio_files: List[Union[str, bytes, os.PathLike]],
    output_files: List[Union[str, bytes, os.PathLike]],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    num_workers: int = 0,
    gpu: Optional[int] = None,
    max_frames: int = ppgs.MAX_INFERENCE_FRAMES,
    legacy_mode: bool = False
) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        audio_files
            The audio files
        output_files
            The .pt files to save PPGs
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        num_workers
            Number of CPU threads for multiprocessing
        gpu
            The index of the GPU to use for inference
        max_frames
            The maximum number of frames on the GPU at once
        legacy_mode
            Use legacy (unchunked) inference
    """
    # Single-threaded
    if num_workers == 0:
        infer_fn = functools.partial(
            from_file_to_file,
            representation=representation,
            checkpoint=checkpoint,
            gpu=gpu,
            legacy_mode=legacy_mode)
        for audio_file, output_file in zip(audio_files, output_files):
            infer_fn(audio_file, output_file)

    # Multi-threaded
    else:

        # Initialize multi-threaded dataloader
        dataloader = ppgs.data.loader(
            audio_files,
            features=['audio', 'length', 'audio_file'],
            num_workers=num_workers // 2,
            max_frames=max_frames)

        # Maintain file correspondence
        output_files = {
            audio_file: output_file
            for audio_file, output_file in zip(audio_files, output_files)}

        # Batch inference
        from_dataloader(
            dataloader=dataloader,
            output_files=output_files,
            representation=representation,
            checkpoint=checkpoint,
            save_workers=num_workers // 2,
            gpu=gpu,
            legacy_mode=legacy_mode
        )


###############################################################################
# Multiprocessing
###############################################################################


def from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    output_files: Dict[
        Union[str, bytes, os.PathLike],
        Union[str, bytes, os.PathLike]],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Union[str, bytes, os.PathLike] = None,
    save_workers: int = 1,
    gpu: Optional[int] = None,
    legacy_mode: bool = False
) -> None:
    """Infer ppgs from a dataloader yielding audio files

    Arguments
        dataloader
            A DataLoader object to do preprocessing for
            the DataLoader must yield batches (audio, length, audio_filename)
        output_files
            A dictionary mapping audio filenames to output filenames
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        save_workers
            The number of worker threads to use for async file saving
        gpu
            The index of the GPU to use for inference
        legacy_mode
            Use legacy (unchunked) inference
    """
    # Setup multiprocessing
    if save_workers == 0:
        pool = contextlib.nullcontext()
    else:
        pool = mp.get_context('spawn').Pool(save_workers)

    try:

        # Setup progress bar
        progress = torchutil.iterator(
            range(0, len(dataloader.dataset)),
            ppgs.CONFIG,
            total=len(dataloader.dataset))

        # Iterate over dataset
        for audios, lengths, audio_files in dataloader:
            frame_lengths = lengths // ppgs.HOPSIZE

            # Preprocess
            if representation == 'wav':
                features = audios
                attention_mask_lengths = lengths
            else:
                features = getattr(
                    ppgs.preprocess,
                    representation
                ).from_audios(
                    audios,
                    lengths,
                    gpu=gpu)
                attention_mask_lengths = frame_lengths

            if features.requires_grad:
                raise ValueError('All representations should be detached')

            # Infer
            result = from_features(
                features=features,
                lengths=attention_mask_lengths,
                representation=representation,
                checkpoint=checkpoint,
                gpu=gpu,
                legacy_mode=legacy_mode)

            # Get output filenames
            filenames = [output_files[file] for file in audio_files]

            # Save to disk
            if save_workers > 0:

                # Asynchronous save
                pool.starmap_async(
                    ppgs.preprocess.save_masked,
                    zip(result.cpu(), filenames, frame_lengths.cpu()))
                while pool._taskqueue.qsize() > 100:
                    time.sleep(1)

            else:

                # Synchronous save
                for ppg_output, filename, new_length in zip(
                    result.cpu(),
                    filenames,
                    frame_lengths.cpu()
                ):
                    ppgs.preprocess.save_masked(
                        ppg_output,
                        filename,
                        new_length)

            # Increment by batch size
            progress.update(len(audios))

    finally:

        # Close progress bar
        progress.close()

        # Maybe shutdown multiprocessing
        if save_workers > 0:
            pool.close()
            pool.join()


###############################################################################
# PPG distance
###############################################################################


def distance(
    ppgX: torch.Tensor,
    ppgY: torch.Tensor,
    reduction: str = 'mean',
    normalize: bool = True,
    exponent: float = ppgs.SIMILARITY_EXPONENT
) -> torch.Tensor:
    """Compute the pronunciation distance between two aligned PPGs

    Arguments
        ppgX
            Input PPG X
            shape=(len(ppgs.PHONEMES), frames)
        ppgY
            Input PPG Y to compare with PPG X
            shape=(len(ppgs.PHONEMES), frames)
        reduction
            Reduction to apply to the output. One of ['mean', 'none', 'sum'].
        normalize
            Apply similarity based normalization
        exponent
            Similarty exponent

    Returns
        Normalized Jenson-shannon divergence between PPGs
    """
    # Handle numerical instability at boundaries
    ppgX = torch.clamp(ppgX, 1e-8, 1 - 1e-8)
    ppgY = torch.clamp(ppgY, 1e-8, 1 - 1e-8)
    if normalize:
        if (
            not hasattr(distance, 'similarity_matrix') or
            distance.device != ppgX.device
        ):
            distance.similarity_matrix = torch.load(
                ppgs.SIMILARITY_MATRIX_PATH
            ).to(device=ppgX.device, dtype=ppgX.dtype)
            distance.device = ppgX.device
        ppgX = torch.mm(distance.similarity_matrix.T ** exponent, ppgX).T
        ppgY = torch.mm(distance.similarity_matrix.T ** exponent, ppgY).T
    else:
        ppgX = ppgX.T
        ppgY = ppgY.T

    # Average in parameter space
    log_average = torch.log((ppgX + ppgY) / 2)

    # Compute KL divergences in both directions
    kl_X = torch.nn.functional.kl_div(
        log_average,
        ppgX,
        reduction='none')
    kl_Y = torch.nn.functional.kl_div(
        log_average,
        ppgY,
        reduction='none')

    # Average KL
    average_kl = (kl_X + kl_Y) / 2
    average_kl[average_kl < 0] = 0
    jsd = torch.sqrt(average_kl)
    jsd = jsd.sum(dim=1)

    # Maybe reduce
    if reduction == 'mean':
        return jsd.mean(dim=0)
    elif reduction == 'none' or reduction is None:
        return jsd
    elif reduction == 'sum':
        return jsd.sum(dim=0)
    raise ValueError(f'Reduction method {reduction} not defined')


###############################################################################
# PPG interpolation
###############################################################################


def interpolate(
    ppgX: torch.Tensor,
    ppgY: torch.Tensor,
    interp: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Linear interpolation

    Arguments
        ppgX
            Input PPG X
            shape=(len(ppgs.PHONEMES), frames)
        ppgY
            Input PPG Y
            shape=(len(ppgs.PHONEMES), frames)
        interp
            Interpolation values
            scalar float OR shape=(frames,)

    Returns
        Interpolated PPGs
        shape=(len(ppgs.PHONEMES), frames)
    """
    return (1. - interp) * ppgX + interp * ppgY


###############################################################################
# PPG sparsification
###############################################################################


def sparsify(
    ppg: torch.Tensor,
    method: str = 'percentile',
    threshold: torch.Tensor = torch.Tensor([0.85])
) -> torch.Tensor:
    """Make phonetic posteriorgrams sparse

    Arguments
        ppg
            Input PPG
            shape=(batch, len(ppgs.PHONEMES), frames)
        method
            Sparsification method. One of ['constant', 'percentile', 'topk'].
        threshold
            In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.

    Returns
        Sparse phonetic posteriorgram
        shape=(batch, len(ppgs.PHONEMES), frames)
    """
    # Threshold either a constant value or a percentile
    if method in ['constant', 'percentile']:
        if method == 'percentile':
            threshold = torch.quantile(ppg, threshold, dim=-2, keepdim=True)
        ppg = torch.where(ppg > threshold, ppg, 0)

    # Take the top n bins
    elif method == 'topk':
        values, indices = ppg.topk(
            threshold,
            dim=-2)
        ppg.zero_()
        for t in range(ppg.shape[-1]):
            ppg[:, indices[..., t], t] = values[..., t]

    # Renormalize after sparsification
    return torch.softmax(torch.log(ppg + 1e-8), -2)


###############################################################################
# Utilities
###############################################################################


def infer(
    features,
    lengths,
    representation='mel',
    checkpoint=None,
    softmax=True,
    legacy_mode=False):
    """Perform model inference"""

    # Skip inference if we want input representations
    if ppgs.REPRESENTATION_KIND == 'latents':
        return features

    # Maybe cache model
    if not hasattr(infer, 'models'): infer.models = {}
    if (
        not hasattr(infer, 'model') or
        infer.checkpoint != checkpoint or
        infer.device_type != features.device.type or
        infer.representation != representation
    ):
        model_key = str(representation) + str(checkpoint)
        if model_key not in infer.models:
            infer.models[model_key] = ppgs.load.model(
                checkpoint=checkpoint,
                representation=representation)
        infer.model = infer.models[model_key]
        infer.checkpoint = checkpoint
        infer.representation = representation
        infer.device_type = features.device.type

    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(features.device)

    # Infer
    with torchutil.inference.context(infer.model):
        if isinstance(infer.model, ppgs.model.Transformer):
            logits = infer.model(features, lengths, legacy_mode=legacy_mode)
        else:
            logits = infer.model(features, lengths)

        # Postprocess
        if softmax:
            return torch.nn.functional.softmax(logits, dim=1)

        return logits


def resample(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    target_rate: Union[int, float] = ppgs.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)


def representation_file_extension():
    if (
        ppgs.REPRESENTATION == ppgs.BEST_REPRESENTATION and
        ppgs.REPRESENTATION_KIND == 'ppg'
    ):
        return '-ppg.pt'
    else:
        if ppgs.REPRESENTATION_KIND == 'ppg':
            return f'-{ppgs.REPRESENTATION}-ppg.pt'
        else:
            return f'-{ppgs.REPRESENTATION}.pt'
