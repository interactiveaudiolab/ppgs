import itertools

import torchaudio
import torchutil

import ppgs


###############################################################################
# Dataset statistics
###############################################################################


def process(datasets):
    """Compute dataset statistics"""
    for dataset in datasets:
        directory = ppgs.CACHE_DIR / dataset

        # Get stems
        stems = list(
            itertools.chain.from_iterable(
                ppgs.load.partition(dataset).values()))

        # Get duration in seconds
        duration = 0.
        for stem in torchutil.iterator(
            stems,
            f'Computing dataset statistics for {dataset}',
            total=len(stems)
        ):
            info = torchaudio.info(directory / f'{stem}.wav')
            duration += info.num_frames / info.sample_rate

        # Convert to hours
        duration /= 3600

        # Report results
        print(
            f'The {dataset} dataset contains {duration:.2f} '
            f'hours across {len(stems)} files')
