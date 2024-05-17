import argparse
import shutil
from pathlib import Path

import ppgs


###############################################################################
# Training CLI
###############################################################################


def main(config, dataset, gpu=None):
    """Train from configuration"""
    # Create output directory
    directory = ppgs.RUNS_DIR / ppgs.CONFIG
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    if config is not None:
        shutil.copyfile(config, directory / config.name)

    # Train
    ppgs.train(dataset, directory, gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default=ppgs.TRAINING_DATASET,
        help='The dataset to train on')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to run training on')
    return parser.parse_args()


main(**vars(parse_args()))
