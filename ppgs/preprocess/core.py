"""core.py - data preprocessing"""

from contextlib import nullcontext
import multiprocessing as mp
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import tqdm

import ppgs
from ppgs.data.disk import stop_if_disk_full
from ppgs.notify import notify_on_finish

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['phonemes', 'wav', 'w2v2fs', 'bottleneck', 'w2v2fb', 'spectrogram', 'mel', 'unfold', 'encodec', 'w2v2ft']


###############################################################################
# Preprocess
###############################################################################

@notify_on_finish('preprocessing')
def datasets(datasets, features=ALL_FEATURES, gpu=None, num_workers=0):
    """Preprocess a dataset

    Arguments
        datasets - List[str]
            The names of the dataset to preprocess
        features - List[str]
            The names of the features to do preprocessing for
        gpu - int
            The gpu to use for preprocessing
        num_workers - int
            The number of worker threads to use  
    """
    for dataset in datasets:
        dataloader = loader(dataset, num_workers//2)
        from_dataloader(dataloader, features, save_workers=(num_workers+1)//2, gpu=gpu)

def from_dataloader(
    dataloader,
    features,
    save_workers=0,
    gpu=None,
    output_dir=None
): #TODO make output_dir work
    """Preprocess from a dataloader

    Arguments
        dataloader - torch.utils.data.DataLoader
            A DataLoader object to do preprocessing for. 
            the DataLoader must yield batches (audio, length, audio_filename)
        features - List[str]
            The names of the features to do preprocessing for
        gpu - int
            The gpu to use for preprocessing
        output_dir - Path
            The directory to place the features
        save_workers - int
            The number of worker threads to use for async file saving
    """
    feature_processors = [ppgs.REPRESENTATION_MAP[f] for f in features]
    iterator: Iterator[Tuple[torch.Tensor, List[Path], torch.Tensor]] = tqdm.tqdm(
        dataloader,
        desc=f'preprocessing {features} for dataset {dataloader.dataset.metadata.name}',
        total=len(dataloader),
        dynamic_ncols=True
    )
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    pool = mp.get_context('spawn').Pool(save_workers) if save_workers > 0 else nullcontext
    with pool, torch.inference_mode():
        for audios, lengths, audio_files in iterator:
            audios = audios.to(device)
            lengths = lengths.to(device)
            for feature, feature_processor in zip(features, feature_processors):
                outputs = feature_processor.from_audios(audios, lengths, gpu=gpu).cpu()
                if feature != 'w2v2ft':
                    new_lengths = lengths // ppgs.HOPSIZE
                else:
                    new_lengths = lengths + ppgs.preprocess.w2v2ft.WINDOW_SIZE - ppgs.preprocess.w2v2ft.HOP_SIZE
                #TODO fix output_dir
                filenames = [audio_file.parent / f'{audio_file.stem}-{feature}.pt' for audio_file in audio_files]
                if isinstance(pool, mp.Pool):
                    pool.starmap_async(save_masked, zip(outputs, filenames, new_lengths.cpu()))
                    while pool._taskqueue.qsize() > 256:
                        time.sleep(1)
                else:
                    map(save_masked, outputs, filenames, new_lengths.cpu())
            stop_if_disk_full()
        pool.close()
        pool.join()


def from_files_to_files(
    audio_files,
    features=ALL_FEATURES,
    num_workers=0,
    output_dir=None,
    gpu=None):
    """Preprocess from files
    Arguments
        audio_files - List[str]
            A list of audio files to process
        features - List[str]
            The names of the features to do preprocessing for
        num_workers - int
            The number of workers to use
        output_dir - Path
            The directory to place the features
        gpu - int
            The gpu to use for preprocessing
    """
    dataloader = loader(audio_files, num_workers//2)
    from_dataloader(dataloader, features, (num_workers+1)//2, gpu, output_dir)
    

def from_audio(audio, representation=None, sample_rate=ppgs.SAMPLE_RATE, config=None, gpu=None):
    """Preprocess audio using given or configured representation"""

    #Cache model/function
    if representation is None:
        representation = ppgs.REPRESENTATION
    try:
        representation_module = ppgs.REPRESENTATION_MAP[representation]
    except KeyError:
        raise ValueError(f'given representation "{representation}" does not exist')
    if not hasattr(from_audio, representation):
        setattr(from_audio, representation, representation_module.from_audio)

    #Compute representation
    return getattr(from_audio, representation)(
        audio, 
        sample_rate=sample_rate,
        config=config,
        gpu=gpu
    )

###############################################################################
# Utilities
###############################################################################

def save_masked(tensor: torch.Tensor, file, length: torch.Tensor):
    if str(tensor.device) != 'cpu' or str(length.device) != 'cpu':
        print('tensors (and lengths) must be on cpu for thread safety', flush=True)
        raise ValueError('tensors (and lengths) must be on cpu for thread safety')
    try:
        sub_tensor = tensor[..., :length].clone()
        torch.save(sub_tensor, file)
    except Exception as e:
        print(f'error saving file {file}: {e}', flush=True)

def loader(sources, loader_workers):
    dataset_object = ppgs.data.Dataset(sources, features=['wav', 'length', 'audio_file'])
    sampler_object = ppgs.data.Sampler(dataset_object)
    collator_object = ppgs.data.Collator(features=['wav', 'length', 'audio_file'])
    return torch.utils.data.DataLoader(
        dataset=dataset_object,
        batch_sampler=sampler_object,
        num_workers=loader_workers,
        pin_memory=True,
        collate_fn=collator_object
    )