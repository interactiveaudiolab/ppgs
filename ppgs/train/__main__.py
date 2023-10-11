import argparse
import shutil
from pathlib import Path

import ppgs


###############################################################################
# Training CLI
###############################################################################


def main(config, dataset):
    """Train from configuration"""
    # Create output directory
    directory = ppgs.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    ppgs.train(dataset, directory)


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
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
