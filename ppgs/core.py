import multiprocessing as mp
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torchaudio
import tqdm
from torch.utils.data import DataLoader

import ppgs
from ppgs.data import aggregate
from ppgs.data.disk import stop_if_disk_full
from ppgs.preprocess import save_masked

path = Union[Path, PathLike, str]

###############################################################################
# API
###############################################################################

def from_features(
    features: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    gpu: int = None) -> torch.Tensor:
    """infer ppgs from input features (e.g. w2v2fb, mel, etc.)

    Arguments:
        features
            The input features to process in the shape BATCH x DIMS x TIME
        lengths
            The lengths of the features
        representation
            The type of features to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            Path to the checkpoint to use
        gpu
            The gpu to use for preprocessing
    Returns:
        ppgs
            A tensor encoding ppgs with shape BATCH x DIMS x TIME
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    if not hasattr(from_features, 'model'):
        from_features.model = ppgs.load.model(checkpoint=checkpoint).to(device)
    with torch.inference_mode(), torch.autocast('cuda' if gpu is not None else 'cpu'):
        return from_features.model(features, lengths)

def from_sources_to_sinks(
    sources: List[path],
    sinks: List[path] = None,
    audio_extensions: List[str] = ['wav'],
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    representation: str = ppgs.REPRESENTATION,
    save_intermediate_features: bool = False,
    gpu: int = None,
    num_workers: int = 1) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        sources
            paths to audio files and/or directories
        sinks
            The one-to-one corresponding 
        extensions
            extensions to glob for in directories
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            Path to the checkpoint to use
        gpu
            The gpu to use for preprocessing  
    """
    if sinks is not None:
        input_files, output_files = aggregate(sources, sinks, source_extensions=audio_extensions, sink_extension=f'-{representation}-ppg.pt')
    else:
        input_files = aggregate(sources, source_extensions=audio_extensions)
        output_files = None
    from_files_to_files(
        input_files,
        output=output_files,
        checkpoint=checkpoint,
        representation=representation,
        save_intermediate_features=save_intermediate_features,
        gpu=gpu,
        num_workers=num_workers
    )

def from_dataloader(
    dataloader: DataLoader,
    output: Union[path, Dict[path, path]] = None,
    representation: str = ppgs.REPRESENTATION,
    save_intermediate_features: bool = False,
    gpu: int = None,
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    save_workers: int = 1) -> None:
    """Infer ppgs from a dataloader yielding audio files

    Arguments
        dataloader
            A DataLoader object to do preprocessing for
            the DataLoader must yield batches (audio, length, audio_filename)
        representation
            The type of features to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
            Due to a limitation with yapecs, only one kind can be processed at once
        save_intermediate_features - bool
            Saves the intermediate features (e.g. Wav2Vec 2.0 latents) in addition to ppgs
        gpu
            The gpu to use for preprocessing
        checkpoint
            Path to the checkpoint to use
        output
            A directory to put output files, or a dictionary mapping audio filenames to output filenames
        save_workers
            The number of worker threads to use for async file saving
    """
    iterator: Iterator[Tuple[torch.Tensor, List[Path], torch.Tensor]] = tqdm.tqdm(
        dataloader,
        desc=f'processing {representation} for dataset {dataloader.dataset.metadata.name}',
        total=len(dataloader),
        dynamic_ncols=True
    )
    if output is not None:
        if isinstance(output, str):
            output = Path(output)
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    with mp.get_context('spawn').Pool(save_workers) as pool:
        with torch.inference_mode(), torch.autocast('cuda' if gpu is not None else 'cpu'):
            for audios, lengths, audio_files in iterator:
                audios = audios.to(device)
                lengths = lengths.to(device)
                feature_processor = ppgs.REPRESENTATION_MAP[representation]
                # torch.cuda.empty_cache()
                # print(torch.cuda.memory_summary(gpu, abbreviated=True))
                features = feature_processor.from_audios(audios, lengths, gpu=gpu)
                new_lengths = lengths // ppgs.HOPSIZE
                ppg_outputs = from_features(features, new_lengths, checkpoint=checkpoint, gpu=gpu)
                if save_intermediate_features:
                    if output is not None:
                        if isinstance(output, dict):
                            raise ValueError('save_intermediate_features is not compatible with passing output files, pass a directory instead')
                        else:
                            filenames = [output / f'{audio_file.stem}-{representation}.pt' for audio_file in audio_files]
                    else:
                        filenames = [audio_file.parent / f'{audio_file.stem}-{representation}.pt' for audio_file in audio_files]
                    pool.starmap_async(save_masked, zip(features.cpu(), filenames, new_lengths.cpu()))
                if output is not None:
                    if isinstance(output, dict):
                        filenames = [output[audio_file] for audio_file in audio_files]
                    else:
                        filenames = [output / f'{audio_file.stem}-{representation}-ppg.pt' for audio_file in audio_files]
                else:
                    filenames = [audio_file.parent / f'{audio_file.stem}-{representation}-ppg.pt' for audio_file in audio_files]
                pool.starmap_async(save_masked, zip(ppg_outputs.cpu(), filenames, new_lengths.cpu()))
                while pool._taskqueue.qsize() > 100:
                    sleep(1)
                stop_if_disk_full()
        pool.close()
        pool.join()

def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    gpu: int = None) -> torch.Tensor:
    """Infer ppgs from audio

    Arguments
        audio
            The batched audio to process in the shape BATCH x 1 x TIME
        lengths
            The lengths of the features
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            Path to the checkpoint to use
        gpu
            The gpu to use for preprocessing

    Returns
        ppgs
            A tensor encoding ppgs with shape BATCH x DIMS x TIME
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    with torch.inference_mode(), torch.autocast(device.type):
        features = ppgs.preprocess.from_audio(audio, representation=representation, sample_rate=sample_rate, gpu=gpu)
        if features.dim() == 2:
            features = features[None]
        # Compute PPGs
        return from_features(features, torch.tensor([features.shape[-1]]).to(device), checkpoint=checkpoint, gpu=gpu)

def from_file(
        file: path,
        representation: str = ppgs.REPRESENTATION,
        checkpoint: path = ppgs.DEFAULT_CHECKPOINT, 
        gpu=None
    ) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        file
            Path to audio file
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            Path to the checkpoint to use
        gpu
            The gpu to use for preprocessing
    
    Returns
        ppgs
            A tensor encoding ppgs with shape 1 x DIMS x TIME
    """
    # Load audio
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    audio = ppgs.load.audio(file).to(device)

    # Compute PPGs
    return from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, representation=representation, checkpoint=checkpoint, gpu=gpu)


def from_file_to_file(
    audio_file: path,
    output_file: path,
    representation: str = ppgs.REPRESENTATION,
    preprocess_only: bool = False,
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    gpu: int =None) -> None:
    """Infer ppg from an audio file and save to a torch tensor file

    Arguments
        audio_file
            Path to audio file
        output_file
            Path to output file (ideally '.pt')
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        preprocess_only
            Shortcut to just doing preprocessing for the given representation
        checkpoint
            Path to the checkpoint to use
        gpu
            The gpu to use for preprocessing  
    """
    # Compute PPGs
    result = from_file(audio_file, representation=representation, preprocess_only=preprocess_only, checkpoint=checkpoint, gpu=gpu).detach().cpu()

    # Save to disk
    torch.save(result, output_file)


def from_files_to_files(
    audio_files: List[path],
    output: Union[List[path], path] = None,
    representation: str = None,
    checkpoint: path = ppgs.DEFAULT_CHECKPOINT,
    save_intermediate_features: bool = False,
    num_workers: int = 1,
    gpu: int = None) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        audio_files
            Path to audio files
        output
            A list of output files or a path to an output directory
            If not provided, ppgs will be stored in same locations as audio files
        representation
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint
            Path to the checkpoint to use
        save_intermediate_features
            Saves the intermediate features (e.g. Wav2Vec 2.0 latents) in addition to ppgs
        gpu
            The gpu to use for preprocessing
    """
    dataloader = ppgs.preprocess.loader(audio_files, num_workers//2)
    if isinstance(output, list): #handle list of output files
        output = {audio_file: output_file for audio_file, output_file in zip(audio_files, output)}
    elif isinstance(output, str): #convert str to Path
        output = Path(output)
        assert output.is_dir(), "If a single output path is provided, it must be a directory"
    from_dataloader(
        dataloader,
        output=output,
        representation=representation,
        checkpoint=checkpoint,
        save_workers=(num_workers+1)//2, 
        save_intermediate_features=save_intermediate_features,
        gpu=gpu)


###############################################################################
# Utilities
###############################################################################

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