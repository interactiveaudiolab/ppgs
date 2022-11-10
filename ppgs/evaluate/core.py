"""core.py - model evaluation"""

import ppgs
import json
import time
import tqdm


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, checkpoint=None, gpu=None):
    """Perform evaluation"""
    
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

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        print(dataset)
        # Setup test dataset
        dataloader = ppgs.data.loader.loader(dataset, 'test', representation=ppgs.REPRESENTATION)
        iterator = tqdm.tqdm(
            dataloader,
            f'Evaluating {ppgs.CONFIG} on {dataset}',
            len(dataloader),
            dynamic_ncols=True
        )

        # Iterate over test set
        for input_ppgs, indices, alignment, word_breaks, audio, stems in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Infer
            logits = ppgs.from_audio(
                audio[0][0], #get only audio, and ignore sample rate (audio[0][1])
                ppgs.SAMPLE_RATE,
                checkpoint=checkpoint,
                # batch_size=2048, #TODO determine why this is here?
                gpu=gpu).T.cpu()

            indices = indices.squeeze()

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
    with open(directory / 'overall.json', 'w') as file:
        json.dump(overall, file, indent=4)
    with open(directory / 'granular.json', 'w') as file:
        json.dump(granular, file, indent=4)

    # Turn off benchmarking
    ppgs.BENCHMARK = False

    # Get benchmarking information
    # benchmark = ppgs.TIMER()
    benchmark = {}
    benchmark['elapsed'] = time.time() - start

    # Get total number of frames, samples, and seconds in test data
    #TODO make better way of accessing loss
    frames = aggregate_metrics.metrics[1].count
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
    with open(directory / 'time.json', 'w') as file:
        json.dump(results, file, indent=4)
