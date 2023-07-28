"""core.py - model evaluation"""

import ppgs
import json
import time
import tqdm
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from ppgs.notify import notify_on_finish
import torch


###############################################################################
# Evaluate
###############################################################################

@notify_on_finish('evaluate')
def datasets(datasets, model_source: Path=None, gpu=None, partition=None):
    """Perform evaluation"""

    model_source = Path(model_source)
    if not model_source.exists():
        raise FileNotFoundError(f'model source \'{model_source}\' does not exist')
    if model_source.is_dir():
        # Get the config file and configure with yapecs
        configs = list(model_source.glob('*.py'))
        assert len(configs) == 1, 'there must be exactly one python file in a run directory'
        config = configs[0]


        checkpoints = list(model_source.glob('*.pt'))
        steps = [int(checkpoint.stem) for checkpoint in checkpoints]
        checkpoint = checkpoints[np.argmax(steps)]
        import pdb; pdb.set_trace()
    else:
        checkpoint = model_source

    with ExitStack() as stack:
    
        # Start benchmarking
        ppgs.BENCHMARK = True
        #TODO restore functionality
        # ppgs.TIMER.reset()
        start = time.time()

        # Containers for results
        overall, granular = {}, {}

        # Per-file metrics
        file_metrics = ppgs.evaluate.Metrics('per-file')

        # Per-dataset metrics
        dataset_metrics = ppgs.evaluate.Metrics('per-dataset')

        # Aggregate metrics over all datasets
        aggregate_metrics = ppgs.evaluate.Metrics('aggregate')

        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Evaluate each dataset
        for dataset in datasets:

            # Reset dataset metrics
            dataset_metrics.reset()

            # Setup test dataset
            ppgs.BATCH_SIZE = 1
            dataloader = ppgs.data.loader.loader(dataset, partition, features=[ppgs.REPRESENTATION, 'length', 'phonemes', 'stem'])
            iterator = tqdm.tqdm(
                dataloader,
                f'Evaluating {ppgs.CONFIG} on {dataset}',
                len(dataloader),
                dynamic_ncols=True
            )

            # Iterate over test set
            for input_ppgs, lengths, indices, stems in iterator:

                # Reset file metrics
                file_metrics.reset()

                input_ppgs = input_ppgs.to(device)
                lengths = lengths.to(device)
                indices = indices.to(device)
                logits = ppgs.from_features(input_ppgs, lengths, checkpoint=checkpoint, gpu=gpu)

                # Update metrics
                file_metrics.update(logits, indices)
                dataset_metrics.update(logits, indices)
                aggregate_metrics.update(logits, indices)

                # Copy results
                stem = stems[0]
                granular[f'{dataset}/{stem}'] = file_metrics()
            overall[dataset] = dataset_metrics()
        overall['aggregate'] = aggregate_metrics()

        # Make output directory
        directory = ppgs.EVAL_DIR / ppgs.CONFIG
        directory.mkdir(exist_ok=True, parents=True)

        # Write to json files
        with open(directory / f'overall-{partition}.json', 'w') as file:
            json.dump(overall, file, indent=4)
        with open(directory / f'granular-{partition}.json', 'w') as file:
            json.dump(granular, file, indent=4)

        # Turn off benchmarking
        ppgs.BENCHMARK = False

        # Get benchmarking information
        # benchmark = ppgs.TIMER()
        benchmark = {}
        benchmark['elapsed'] = time.time() - start

        # Get total number of frames, samples, and seconds in test data
        #TODO make better way of accessing loss
        frames = aggregate_metrics.metrics[2].count
        #TODO check this
        # samples = ppgs.convert.frames_to_samples(frames)
        samples = int(frames * ppgs.HOPSIZE)
        # TODO check this
        # seconds = ppgs.convert.frames_to_seconds(frames)
        seconds = float(samples / ppgs.SAMPLE_RATE)

        # Format benchmarking results
        results = {
            key: {
                'real-time-factor': seconds / value,
                'samples': samples,
                'samples-per-second': samples / value,
                'total': value
            } for key, value in benchmark.items()}

        # Write benchmarking information
        with open(directory / f'time-{partition}.json', 'w') as file:
            json.dump(results, file, indent=4)
