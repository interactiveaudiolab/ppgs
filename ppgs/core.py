import torch
import torchaudio
import tqdm
from pathlib import Path
import ppgs
from typing import List, Iterator, Tuple
from ppgs.data import aggregate
from ppgs.preprocess import save_masked
from ppgs.data.disk import stop_if_disk_full
import multiprocessing as mp
from time import sleep

###############################################################################
# API
###############################################################################

def from_features(
    features: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    gpu=None
):
    """infer ppgs from input features (e.g. w2v2fb, mel, etc.)

    Arguments
        features - torch.Tensor
            The input features to process in the shape BATCH x DIMS x TIME
        lengths - torch.Tensor
            The lengths of the features
        representation - str
            The type of features to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint - str
            Path to the checkpoint to use
        gpu - int
            The gpu to use for preprocessing
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    if not hasattr(from_features, 'model'):
        from_features.model = ppgs.load.model(checkpoint=checkpoint).to(device)
    with torch.inference_mode(), torch.autocast('cuda' if gpu is not None else 'cpu'):
        return from_features.model(features, lengths)

def from_sources_to_files(
    sources,
    output_dir=None,
    extensions=['wav'],
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    representation=ppgs.REPRESENTATION,
    save_intermediate_features=False,
    gpu=None,
    num_workers=1
):
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        sources - List[str]
            paths to audio files and/or directories
        output_dir - Path
            The directory to place the ppgs
            If not provided, ppgs will be stored in same locations as audio files
        extensions - List[str]
            extensions to glob for in directories
        representation - str
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint - str
            Path to the checkpoint to use
        gpu - int
            The gpu to use for preprocessing  
    """
    files = aggregate(sources, extensions)
    from_files_to_files(
        files,
        output_dir=output_dir,
        checkpoint=checkpoint,
        representation=representation,
        save_intermediate_features=save_intermediate_features,
        gpu=gpu,
        num_workers=num_workers
    )

def from_dataloader(
    dataloader,
    representation,
    save_intermediate_features=False,
    gpu=None,
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    output_dir=None,
    save_workers=1
):
    """Infer ppgs from a dataloader yielding audio files

    Arguments
        dataloader - torch.utils.data.DataLoader
            A DataLoader object to do preprocessing for
            the DataLoader must yield batches (audio, length, audio_filename)
        representation - str
            The type of features to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
            Due to a limitation with yapecs, only one kind can be processed at once
        save_intermediate_features - bool
            Saves the intermediate features (e.g. Wav2Vec 2.0 latents) in addition to ppgs
        gpu - int
            The gpu to use for preprocessing
        checkpoint - str
            Path to the checkpoint to use
        output_dir - Union[str, Path]
            The directory to place the ppgs (and intermediate features)
        save_workers - int
            The number of worker threads to use for async file saving
    """
    iterator: Iterator[Tuple[torch.Tensor, List[Path], torch.Tensor]] = tqdm.tqdm(
        dataloader,
        desc=f'processing {representation} for dataset {dataloader.dataset.metadata.name}',
        total=len(dataloader),
        dynamic_ncols=True
    )
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
                    if output_dir is not None:
                        filenames = [output_dir / f'{audio_file.stem}-{representation}.pt' for audio_file in audio_files]
                    else:
                        filenames = [audio_file.parent / f'{audio_file.stem}-{representation}.pt' for audio_file in audio_files]
                    pool.starmap_async(save_masked, zip(features.cpu(), filenames, new_lengths.cpu()))
                if output_dir is not None:
                    filenames = [output_dir / f'{audio_file.stem}-{representation}-ppg.pt' for audio_file in audio_files]
                else:
                    filenames = [audio_file.parent / f'{audio_file.stem}-{representation}-ppg.pt' for audio_file in audio_files]
                pool.starmap_async(save_masked, zip(ppg_outputs.cpu(), filenames, new_lengths.cpu()))
                while pool._taskqueue.qsize() > 100:
                    sleep(1)
                stop_if_disk_full()
        pool.close()
        pool.join()

def from_audio(
    audio,
    sample_rate,
    representation=ppgs.REPRESENTATION,
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    gpu=None):
    """Infer ppgs from audio

    Arguments
        audio - torch.Tensor
            the batched audio to process in the shape BATCH x 1 x TIME
        lengths - torch.Tensor
            The lengths of the features
        representation - str
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint - str
            Path to the checkpoint to use
        gpu - int
            The gpu to use for preprocessing  
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    with torch.inference_mode(), torch.autocast(device.type):
        features = ppgs.preprocess.from_audio(audio, representation=representation, sample_rate=sample_rate, gpu=gpu)
        if features.dim() == 2:
            features = features[None]
        # Compute PPGs
        return from_features(features, torch.tensor([features.shape[-1]]), checkpoint=checkpoint, gpu=gpu)

def from_file(
        file,
        representation=ppgs.REPRESENTATION,
        preprocess_only=False, 
        checkpoint=ppgs.DEFAULT_CHECKPOINT, 
        gpu=None
    ):
    """Infer ppgs from an audio file

    Arguments
        file - str
            Path to audio file
        representation - str
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        preprocess_only - bool
            Shortcut to just doing preprocessing for the given representation
        checkpoint - str
            Path to the checkpoint to use
        gpu - int
            The gpu to use for preprocessing  
    """
    # Load audio
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    audio = ppgs.load.audio(file).to(device)

    # Compute PPGs
    return from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, representation=representation, preprocess_only=preprocess_only, checkpoint=checkpoint, gpu=gpu)


def from_file_to_file(
    audio_file,
    output_file,
    representation=ppgs.REPRESENTATION,
    preprocess_only=False,
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    gpu=None):
    """Infer ppg from an audio file and save to a torch tensor file

    Arguments
        audio_file - str
            Path to audio file
        output_file - str
            Path to output file (ideally '.pt')
        representation - str
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        preprocess_only - bool
            Shortcut to just doing preprocessing for the given representation
        checkpoint - str
            Path to the checkpoint to use
        gpu - int
            The gpu to use for preprocessing  
    """
    # Compute PPGs
    result = from_file(audio_file, representation=representation, preprocess_only=preprocess_only, checkpoint=checkpoint, gpu=gpu).detach().cpu()

    # Save to disk
    torch.save(result, output_file)


def from_files_to_files(
    audio_files,
    output_dir=None,
    representation=None,
    checkpoint=ppgs.DEFAULT_CHECKPOINT,
    save_intermediate_features=False,
    num_workers=1,
    gpu=None):
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        audio_file - List[str]
            Path to audio files
        output_dir - Path
            The directory to place the ppgs
            If not provided, ppgs will be stored in same locations as audio files
        representation - str
            The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
        checkpoint - str
            Path to the checkpoint to use
        save_intermediate_features - bool
            Saves the intermediate features (e.g. Wav2Vec 2.0 latents) in addition to ppgs
        gpu - int
            The gpu to use for preprocessing
    """
    dataloader = ppgs.preprocess.loader(audio_files, num_workers//2)
    from_dataloader(
        dataloader,
        representation=representation,
        checkpoint=checkpoint,
        save_workers=(num_workers+1)//2, 
        save_intermediate_features=save_intermediate_features,
        gpu=gpu,
        output_dir=output_dir)


###############################################################################
# Utilities
###############################################################################

def resample(audio, sample_rate, target_rate=ppgs.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)