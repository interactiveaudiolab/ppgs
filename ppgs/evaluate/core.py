"""core.py - model evaluation"""

import ppgs
import json
import time
import tqdm


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, config, gpu=None):
    """Perform evaluation"""
    if representation is None:
        representation = ppgs.REPRESENTATION


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
        for audio, bins, pitch, voiced, stem in iterator:

            # Reset file metrics
            file_metrics.reset()

            # Infer
            # _, _, logits = ppgs.from_audio(
            #     audio[0],
            #     ppgs.SAMPLE_RATE,
            #     model=ppgs.MODEL,
            #     checkpoint=checkpoint,
            #     batch_size=2048,
            #     gpu=gpu)


            # Update metrics
            file_metrics.update(logits, bins, pitch, voiced)
            dataset_metrics.update(logits, bins, pitch, voiced)
            aggregate_metrics.update(logits, bins, pitch, voiced)

            # Copy results
            granular[f'{dataset}/{stem[0]}'] = file_metrics()
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
    frames = aggregate_metrics.loss.count
    samples = ppgs.convert.frames_to_samples(frames)
    seconds = ppgs.convert.frames_to_seconds(frames)

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
