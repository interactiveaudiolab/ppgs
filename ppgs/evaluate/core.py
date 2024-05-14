import json
from matplotlib.figure import Figure

import torch
import torchutil

import ppgs


###############################################################################
# Evaluate
###############################################################################


@torchutil.notify('evaluate')
def datasets(datasets, gpu=None, checkpoint=None):
    """Perform evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Get model checkpoint
    if checkpoint is None:
        checkpoint = torchutil.checkpoint.latest_path(
            ppgs.RUNS_DIR / ppgs.CONFIG)

    # Containers for results
    results = {}

    # Per-dataset metrics
    dataset_metrics = ppgs.evaluate.Metrics()

    # Aggregate metrics over all datasets
    aggregate_metrics = ppgs.evaluate.Metrics(include_figures=True)

    # Evaluate each dataset
    for dataset in datasets:

        # Reset dataset metrics
        dataset_metrics.reset()

        # Setup test dataset
        dataloader = ppgs.data.loader(dataset, 'test')

        # Iterate over test set
        for input_features, indices, lengths in torchutil.iterator(
            dataloader,
            f'Evaluating {ppgs.CONFIG} on {dataset}',
            total=len(dataloader)
        ):
            # Infer PPGs
            logits = ppgs.from_features(
                input_features,
                lengths,
                checkpoint=checkpoint,
                gpu=gpu,
                softmax=False)

            # Update metrics
            indices = indices.to(device)
            dataset_metrics.update(logits, indices)
            aggregate_metrics.update(logits, indices)
        results[dataset] = dataset_metrics()
    results['aggregate'] = aggregate_metrics()

    # Make output directory
    directory = ppgs.EVAL_DIR / ppgs.CONFIG
    directory.mkdir(exist_ok=True, parents=True)

    # Save to disk
    save(results, f'overall', directory)


###############################################################################
# Utilities
###############################################################################


def save(results, name, directory, save_json=True):
    """Save metrics and maybe figures"""
    fig_dir = directory / name
    fig_dir.mkdir(exist_ok=True, parents=True)
    for metric, value in list(results.items()):

        # Recursively save and remove non-json-serializable results
        if isinstance(value, dict):
            save(value, name, directory, save_json=False)

        # Save and remove figure
        elif isinstance(value, Figure):
            value.savefig(
                fig_dir / f'{metric.replace("/", "-")}.jpg',
                bbox_inches='tight',
                pad_inches=0)
            value.savefig(
                fig_dir / f'{metric.replace("/", "-")}.pdf',
                bbox_inches='tight',
                pad_inches=0)
            del results[metric]

        # Save and remove tensor
        elif isinstance(value, torch.Tensor) and value.dim() >= 1:
            torch.save(value, fig_dir / f'{metric.replace("/", "-")}.pt')
            del results[metric]

    # Save json-serializable results
    if save_json:
        with open(directory / f'{name}.json', 'w') as file:
            json.dump(results, file, indent=4)
