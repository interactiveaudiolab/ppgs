import contextlib
import functools
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import tqdm

import ppgs


###############################################################################
# Application programming interface
###############################################################################


def from_features(
    features: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None,
    softmax: bool = True) -> torch.Tensor:
    """Infer ppgs from input features (e.g. w2v2fb, mel, etc.)

    Arguments
        features
            The input features to process in the shape BATCH x DIMS x TIME
        lengths
            The lengths of the features
        representation
            The type of features to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference
        softmax
            Whether to apply softmax normalization to the inferred logits

    Returns
        ppgs
            A tensor encoding ppgs with shape BATCH x DIMS x TIME
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Maybe load and cache codebook
    if not hasattr(from_features, 'frontend') and ppgs.FRONTEND is not None:
        from_features.frontend = ppgs.FRONTEND(device)

    # Codebook lookup
    if hasattr(from_features, 'frontend'):
        preprocess_context = (
            torch.inference_mode if ppgs.MODEL == 'W2V2FC'
            else inference_context)
        with preprocess_context():
            features = from_features.frontend(features)

    # Infer
    return infer(features.to(device), lengths.to(device), checkpoint, softmax)


def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    gpu: int = None) -> torch.Tensor:
    """Infer ppgs from audio

    Arguments
        audio
            The batched audio to process in the shape BATCH x 1 x TIME
        lengths
            The lengths of the features
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            A tensor encoding ppgs with shape BATCH x DIMS x TIME
    """
    # Preprocess
    features = ppgs.preprocess.from_audio(audio, sample_rate, gpu)

    # Get length in frames
    length = torch.tensor([features.shape[-1]], dtype=torch.long)

    # Infer
    return from_features(features, length, checkpoint, gpu)


def from_file(
    file: Union[str, bytes, os.PathLike],
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        file
            The audio file
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            A tensor encoding ppgs with shape 1 x DIMS x TIME
    """
    # Load audio
    audio = ppgs.load.audio(file)

    # Compute PPGs
    return from_audio(audio, ppgs.SAMPLE_RATE, checkpoint, gpu)


def from_file_to_file(
    audio_file: Union[str, bytes, os.PathLike],
    output_file: Union[str, bytes, os.PathLike],
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> None:
    """Infer ppg from an audio file and save to a torch tensor file

    Arguments
        audio_file
            The audio file
        output_file
            The .pt file to save PPGs
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference
    """
    # Compute PPGs
    result = from_file(audio_file, checkpoint, gpu)

    # Save to disk
    torch.save(result.detach().cpu(), output_file)


def from_files_to_files(
    audio_files: List[Union[str, bytes, os.PathLike]],
    output_files: List[Union[str, bytes, os.PathLike]],
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    num_workers: int = 8,
    gpu: Optional[int] = None) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        audio_files
            The audio files
        output_files
            The .pt files to save PPGs
        checkpoint
            The checkpoint file
        num_workers
            Number of CPU threads for multiprocessing
        gpu
            The index of the GPU to use for inference
    """
    # Single-threaded
    if num_workers == 0:
        infer_fn = functools.partial(
            from_file_to_file,
            checkpoint=checkpoint,
            gpu=gpu)
        for audio_file, output_file in zip(audio_files, output_files):
            infer_fn(audio_file, output_file)

    # Multi-threaded
    else:

        # Initialize multi-threaded dataloader
        dataloader = ppgs.data.loader(
            audio_files,
            features=['wav', 'length', 'audio_file'],
            num_workers=num_workers // 2)

        # Maintain file correspondence
        output_files = {
            audio_file: output_file
            for audio_file, output_file in zip(audio_files, output_files)}

        # Batch inference
        from_dataloader(
            dataloader,
            output_files,
            checkpoint,
            num_workers // 2,
            gpu)


def from_paths_to_paths(
    input_paths: List[Union[str, bytes, os.PathLike]],
    output_paths: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    extensions: Optional[List[str]] = None,
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    num_workers: int = 8,
    gpu: Optional[int] = None) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        input_paths
            Paths to audio files and/or directories
        output_paths
            The one-to-one corresponding outputs
        extensions
            Extensions to glob for in directories
        checkpoint
            The checkpoint file
        num_workers
            Number of CPU threads for multiprocessing
        gpu
            The index of the GPU to use for inference
    """
    if output_paths is not None:
        input_files, output_files = ppgs.data.aggregate(
            input_paths,
            output_paths,
            extensions,
            f'-{ppgs.REPRESENTATION}-ppg.pt')
    else:
        input_files = ppgs.data.aggregate(
            input_paths,
            source_extensions=extensions)
        output_files = None
    from_files_to_files(
        input_files,
        output_files,
        checkpoint,
        gpu=gpu,
        num_workers=num_workers)


###############################################################################
# Multiprocessing
###############################################################################


def from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    output_files: Dict[
        Union[str, bytes, os.PathLike],
        Union[str, bytes, os.PathLike]],
    checkpoint: Union[str, bytes, os.PathLike] = ppgs.DEFAULT_CHECKPOINT,
    save_workers: int = 1,
    gpu: Optional[int] = None) -> None:
    """Infer ppgs from a dataloader yielding audio files

    Arguments
        dataloader
            A DataLoader object to do preprocessing for
            the DataLoader must yield batches (audio, length, audio_filename)
        output_files
            A dictionary mapping audio filenames to output filenames
        checkpoint
            The checkpoint file
        save_workers
            The number of worker threads to use for async file saving
        gpu
            The index of the GPU to use for inference
    """
    # Setup multiprocessing
    if save_workers == 0:
        pool = contextlib.nullcontext()
    else:
        pool = mp.get_context('spawn').Pool(save_workers)
    with pool:

        # Iterate over dataset
        message = (
            f'Processing {ppgs.REPRESENTATION} for '
            f'dataset {dataloader.dataset.metadata.name}')
        for audios, lengths, audio_files in iterator(
            dataloader,
            message,
            total=len(dataloader)
        ):
            frame_lengths = lengths // ppgs.HOPSIZE

            # Preprocess
            if ppgs.REPRESENTATION == 'wav':
                features = audios
                attention_mask_lengths = lengths
            else:
                features = getattr(
                    ppgs.preprocess,
                    ppgs.REPRESENTATION
                ).from_audios(
                    audios,
                    lengths,
                    gpu=gpu)
                attention_mask_lengths = frame_lengths

            # Infer
            result = from_features(
                features,
                attention_mask_lengths,
                checkpoint,
                gpu)

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


###############################################################################
# Utilities
###############################################################################


def aggregate(
    sources: List[Union[str, bytes, os.PathLike]],
    sinks: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    source_extensions: Optional[str] = None,
    sink_extension: str = '.pt'):
    """
    Aggregates lists of input and output directories and files into two lists
    of files, using the provided extension to glob directories.
    """
    # Standardize extensions
    source_extensions = set([
        '.' + ext if '.' not in ext else ext
        for ext in source_extensions])
    sink_extension = (
        '.' + sink_extension if '.' not in sink_extension
        else sink_extension)

    if sinks is None:

        # Get sources as a list of files
        source_files = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                for extension in source_extensions:
                    source_files += list(source.rglob(f'*{extension}'))
            else:
                source_files.append(source)

        # Sink files are source files with sink extension
        sink_files = [file.with_suffix(sink_extension) for file in source_files]

    else:

        # Get sources and sinks as file lists
        source_files, sink_files = [], []
        for source, sink in zip(sources, sinks):
            source = Path(source)
            sink = Path(sink)

            # Handle input directory
            if source.is_dir():
                if not sink.is_dir():
                    raise RuntimeError(
                        f'For input directory {source}, corresponding '
                        f'output {sink} is not a directory')
                for extension in source_extensions:
                    source_files += list(source.rglob(f'*{extension}'))

                # Ensure one-to-one
                source_stems = [file.stem for file in source_files]
                if not len(source_stems) == len(set(source_stems)):
                    raise ValueError(
                        'Two or more files have the same '
                        'stem with different extensions')

                # Get corresponding output files
                sink_files += [sink / (file.stem + sink_extension) for file in source_files]

            # Handle input file
            else:
                if sink.is_dir():
                    raise RuntimeError(
                        f'For input file {source}, corresponding '
                        f'output {sink} is a directory')
                source_files.append(source)
                sink_files.append(sink)

        return source_files, sink_files


def infer(features, lengths, checkpoint=ppgs.DEFAULT_CHECKPOINT, softmax=True):
    """Perform model inference"""
    # Maybe cache model
    if (
        not hasattr(infer, 'model') or
        infer.checkpoint != checkpoint or
        infer.device_type != features.device.type
    ):
        infer.model = ppgs.load.model(checkpoint=checkpoint)
        infer.checkpoint = checkpoint
        infer.device_type = features.device.type

    # Move model to correct device (no-op if devices are the same)
    infer.model = infer.model.to(features.device)

    # Infer
    with inference_context(infer.model):
        logits = infer.model(features, lengths)

        # Postprocess
        if softmax:
            return torch.nn.functional.softmax(logits, dim=1)
        return logits


@contextlib.contextmanager
def inference_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation; turn on mixed precision
    with torch.inference_mode(), torch.autocast(device_type):
        yield

    # Prepare model for training
    model.train()


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


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
