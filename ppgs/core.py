import contextlib
import functools
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from itertools import repeat

import torch
import torchaudio
import tqdm

import ppgs


###############################################################################
# Application programming interface
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    checkpoint: Union[str, bytes, os.PathLike] = None,
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


def from_features(
    features: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint: Union[str, bytes, os.PathLike] = None,
    gpu: Optional[int] = None,
    softmax: bool = True) -> torch.Tensor:
    """Infer ppgs from input features (e.g. w2v2fb, mel, etc.)

    Arguments
        features
            The input features to process in the shape BATCH x DIMS x TIME
        lengths
            The lengths of the features
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


def from_file(
    file: Union[str, bytes, os.PathLike],
    checkpoint: Union[str, bytes, os.PathLike] = None,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        file
            The audio file
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
    checkpoint: Union[str, bytes, os.PathLike] = None,
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
    checkpoint: Union[str, bytes, os.PathLike] = None,
    num_workers: int = 8,
    gpu: Optional[int] = None,
    max_frames: int = ppgs.MAX_INFERENCE_FRAMES) -> None:
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
        max_frames
            The maximum number of frames on the GPU at once
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
            features=['audio', 'length', 'audio_file'],
            num_workers=num_workers // 2,
            max_frames=max_frames)

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
    checkpoint: Union[str, bytes, os.PathLike] = None,
    num_workers: int = 8,
    gpu: Optional[int] = None,
    max_frames: int = ppgs.MAX_INFERENCE_FRAMES) -> None:
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
        max_frames
            The maximum number of frames on the GPU at once
    """
    if output_paths is not None:
        input_files, output_files = aggregate(
            input_paths,
            sinks=output_paths,
            source_extensions=extensions,
            sink_extension=f'-{ppgs.REPRESENTATION}-ppg.pt')
    else:
        input_files, output_files = aggregate(
            input_paths,
            source_extensions=extensions)
    from_files_to_files(
        input_files,
        output_files,
        checkpoint,
        gpu,
        num_workers,
        max_frames)


###############################################################################
# Multiprocessing
###############################################################################


def from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    output_files: Dict[
        Union[str, bytes, os.PathLike],
        Union[str, bytes, os.PathLike]],
    checkpoint: Union[str, bytes, os.PathLike] = None,
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

    try:

        # Setup progress bar
        progress = iterator(
            range(0, len(dataloader.dataset)),
            f'Inferring PPGs',
            total=len(dataloader.dataset))

        # Iterate over dataset
        for audios, lengths, audio_files in dataloader:
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

            # Increment by batch size
            progress.update(len(audios))

    finally:

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
    reduction: Optional[str] = 'mean',
    normalize: Optional[bool] = True
) -> torch.Tensor:
    """Compute the pronunciation distance between two aligned PPGs

    Arguments
        ppgX
            Input PPG X
        ppgY
            Input PPG Y to compare with PPG X
        reduction
            Reduction to apply to the output. One of ['mean', 'none', 'sum'].
        normalize
            Apply similarity based normalization

    Returns
        Normalized Jenson-shannon divergence between PPGs
    """
    # Handle numerical instability at zero
    ppgX = torch.clamp(ppgX, 1e-9)
    ppgY = torch.clamp(ppgY, 1e-9)

    assert ppgX.device == ppgY.device, 'ppgs in distance computation must be on the same device'

    if normalize:
        if not hasattr(distance, 'similarity_matrix'):
            distance.similarity_matrix = torch.load(ppgs.SIMILARITY_MATRIX_PATH).to(ppgX.device)
            distance.device = ppgX.device
        if ppgX.device != distance.device:
            distance.similarity_matrix = distance.similarity_matrix.to(ppgX.device)
        ppgX = torch.mm(distance.similarity_matrix.T ** 1, ppgX.T).T
        ppgY = torch.mm(distance.similarity_matrix.T ** 1, ppgY.T).T

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

    # Sum reduction
    kl_X = kl_X.sum(dim=-1)
    kl_Y = kl_Y.sum(dim=-1)

    # Average KL
    average_kl = (kl_X + kl_Y) / 2
    average_kl[average_kl < 0] = 0
    jsd = torch.sqrt(average_kl)

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
    interp: torch.Tensor
) -> torch.Tensor:
    """Spherical linear interpolation

    Arguments
        ppgX
            Input PPG X
        ppgY
            Input PPG Y
        interp
            Interpolation values

    Returns
        Interpolated PPGs
    """
    omega = torch.acos((ppgX * ppgY).sum(1))
    sin_omega = torch.sin(omega)
    return (
        (torch.sin((1. - interp) * omega) / sin_omega).unsqueeze(1) * ppgX +
        (torch.sin(interp * omega) / sin_omega).unsqueeze(1) * ppgY)


###############################################################################
# Utilities
###############################################################################


def aggregate(
    *sources: List[Union[str, bytes, os.PathLike]],
    sinks: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    source_extensions: Optional[str] = None,
    sink_extension: str = '.pt'):
    """
    Aggregates lists of input and output directories and files into two lists
    of files, using the provided extension to glob directories.
    """
    if len(sources) == 0:
        raise ValueError('At least one source list must be provided')
    elif len(sources) == 1:
        assert sources[0] is not None, 'At least one source list must be provieded, but found None'
    else:
        assert source_extensions is None

    sources = list(sources)
    for i in range(0, len(sources)):
        if sources[i] is None:
            sources[i] = repeat(None)

    # Standardize extensions
    if source_extensions is not None:
        source_extensions = set([
            '.' + ext if '.' not in ext else ext
            for ext in source_extensions])
    else:
        source_extensions = []
    sink_extension = (
        '.' + sink_extension if '.' not in sink_extension
        else sink_extension)
    

    lengths = set()
    for source_list in sources:
        lengths.add(len(source_list))
    
    assert len(lengths) == 1, 'all source lists must have the same lengths'

    if sinks is not None:
        assert len(sinks) == len(sources[0]), 'sinks must have the same length as the source lists'

    if sinks is None:

        # Get sources as a list of files

        source_files = [[] for _ in sources]
        for source_tuple in zip(*sources):
            source_paths = [Path(source) for source in source_tuple]
            source_is_dir = list(set([source.is_dir() for source in source_paths]))
            if len(source_is_dir) > 1 and True in source_is_dir:
                raise ValueError('cannot handle directories when more than one input list is given')
            elif len(source_is_dir) == 1 and source_is_dir[0]:
                for extension in source_extensions:
                    source_files[0] += list(source_paths[0].rglob(f'*{extension}'))
            else:
                for i, source in enumerate(source_paths):
                    source_files[i].append(source)

        # Sink files are source files with sink extension
        sink_files = [
            file.with_suffix(sink_extension) for file in source_files]

    else:

        # Get sources and sinks as file lists
        source_files, sink_files = [[] for _ in sources], []
        for source_tuple, sink in zip(zip(*sources), sinks):
            source_paths = [Path(source) for source in source_tuple]
            source_is_dir = list(set([source.is_dir() for source in source_paths]))
            sink = Path(sink)

            if len(source_is_dir) > 1 and True in source_is_dir:
                raise ValueError('cannot handle directories when more than one input list is given')

            # Handle input directory (only if one source list)
            if True in source_is_dir:
                if not sink.is_dir():
                    raise RuntimeError(
                        f'For input tuple {source_tuple}, corresponding '
                        f'output {sink} is not a directory')
                for extension in source_extensions:
                    source_files[0] += list(source_tuple[0].rglob(f'*{extension}'))

                # Ensure one-to-one
                source_stems = [file.stem for file in source_files[0]]
                if not len(source_stems) == len(set(source_stems)):
                    raise ValueError(
                        'Two or more files have the same '
                        'stem with different extensions')

                # Get corresponding output files
                sink_files += [
                    sink / (file.stem + sink_extension)
                    for file in source_files]

            # Handle input file(s)
            else:
                if sink.is_dir():
                    raise RuntimeError(
                        f'For input file {source}, corresponding '
                        f'output {sink} is a directory')
                for i, source in enumerate(source_tuple):
                    source_files[i].append(source)
                sink_files.append(sink)

    if len(source_files) == 1:
        source_files = source_files[0]

    return source_files, sink_files


def infer(features, lengths, checkpoint=None, softmax=True):
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
