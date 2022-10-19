import argparse
import shutil
from pathlib import Path

import ppgs


###############################################################################
# Entry point
###############################################################################


def main(config, dataset, gpus=None):
    # Create output directory
    directory = ppgs.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    ppgs.train.run(
        dataset,
        directory,
        directory,
        directory,
        gpus)

    # Evaluate
    ppgs.evaluate.datasets([dataset], directory, gpus)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        default=ppgs.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='arctic',
        help='The dataset to train on')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
