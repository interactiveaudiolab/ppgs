import contextlib
import multiprocessing as mp
import time

import torch
import torchutil

import ppgs


###############################################################################
# Preprocess
###############################################################################


@torchutil.notify('preprocess')
def datasets(
    datasets=ppgs.DATASETS,
    representations=None,
    gpu=None,
    num_workers=0,
    partition=None):
    """Preprocess a dataset

    Arguments
        datasets
            The names of the dataset to preprocess
        representations
            The names of the representations to do preprocessing for
        gpu
            The gpu to use for preprocessing
        num_workers
            The number of worker threads to use
        partition
            The partition to preprocess. Default (None) uses all partitions.
    """
    if representations is None:
        representations = [ppgs.REPRESENTATION]

    for dataset in datasets:

        try:

            # Setup multiprocessed dataloader
            dataloader = ppgs.data.loader(
                dataset,
                partition,
                features=['audio', 'length', 'audio_file'],
                num_workers=num_workers // 2,
                max_frames=ppgs.MAX_PREPROCESS_FRAMES)

        except ValueError:

            # Empty partition
            continue

        output = {
            file: f'{file.parent}/{file.stem}' + '-{}.pt'
            for _, _, files in dataloader for file in files}
        from_dataloader(
            dataloader,
            representations,
            output,
            num_workers=(num_workers + 1) // 2,
            gpu=gpu)


def from_files_to_files(
    audio_files,
    output_files,
    representations=ppgs.REPRESENTATION,
    num_workers=0,
    gpu=None):
    """Preprocess from files

    Arguments
        audio_files
            A list of audio files to process
        output_files
            A list of output files to use to save representations
        representations
            The names of the representations to do preprocessing for
        num_workers
            The number of worker threads to use
        gpu
            The gpu to use for preprocessing
    """
    # Setup dataloader
    dataloader = ppgs.data.loader(
        audio_files,
        features=['audio', 'length', 'audio_file'],
        num_workers=num_workers//2,
        max_frames=ppgs.MAX_PREPROCESS_FRAMES)
    from_dataloader(
        dataloader,
        representations,
        dict(zip(audio_files, output_files)),
        num_workers=(num_workers+1)//2,
        gpu=gpu)


###############################################################################
# Utilities
###############################################################################


def from_dataloader(loader, representations, output, num_workers=0, gpu=None):
    """Preprocess from a dataloader

    Arguments
        loader
            A Pytorch DataLoader yielding batches of (audio, length, filename)
        representations
            The names of the representations to do preprocessing for
        output
            A dictionary mapping audio filenames to output filenames
        num_workers
            The number of worker threads to use for async file saving
        gpu
            The gpu to use for preprocessing
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup multiprocessing
    if num_workers == 0:
        pool = contextlib.nullcontext()
    else:
        pool = mp.get_context('spawn').Pool(num_workers)

    try:

        with torch.inference_mode():

            # Batch preprocess
            for audios, lengths, audio_files in torchutil.iterator(
                loader,
                f'Preprocessing {", ".join(representations)} '
                f'for {loader.dataset.metadata.name}',
                total=len(loader)
            ):
                # Copy to device
                audios = audios.to(device)
                lengths = lengths.to(device)

                for representation in representations:

                    # Preprocess
                    outputs = getattr(
                        ppgs.preprocess,
                        representation
                    ).from_audios(audios, lengths, gpu=gpu).cpu()

                    # Get length in frames
                    frame_lengths = lengths // ppgs.HOPSIZE

                    # Get output filenames
                    filenames = []
                    for file in audio_files:
                        output_file = output[file]
                        if '{}' in output_file:
                            filenames.append(
                                output_file.format(representation))
                        else:
                            filenames.append(output_file)

                    if num_workers == 0:

                        # Synchronous save
                        for latent_output, filename, new_length in zip(
                            outputs.cpu(),
                            filenames,
                            frame_lengths.cpu()
                        ):
                            save_masked(latent_output, filename, new_length)
                    else:

                        # Asynchronous save
                        pool.starmap_async(
                            save_masked,
                            zip(outputs, filenames, frame_lengths.cpu()))

                        # Wait if the queue is full
                        while pool._taskqueue.qsize() > 256:
                            time.sleep(1)

    finally:

        # Shutdown multiprocessing
        if num_workers > 0:
            pool.close()
            pool.join()


def from_audio(
    audio,
    representation=ppgs.REPRESENTATION,
    sample_rate=ppgs.SAMPLE_RATE,
    gpu=None
):
    """Preprocess audio"""
    audio = ppgs.resample(audio, sample_rate)

    if representation is None:
        representation = ppgs.REPRESENTATION

    # Compute representation
    with torch.autocast('cuda' if gpu is not None else 'cpu'):
        features = getattr(ppgs.preprocess, representation).from_audio(
            audio,
            sample_rate=ppgs.SAMPLE_RATE,
            gpu=gpu)

        if features.dim() == 2:
            features = features[None]

        return features


def save_masked(tensor, file, length):
    """Save masked tensor"""
    torch.save(tensor[..., :length].clone(), file)
