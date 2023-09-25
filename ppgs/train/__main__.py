import argparse
from pathlib import Path

import ppgs


###############################################################################
# Training CLI
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='charsiu',
        help='The dataset to train on')
    return parser.parse_args()


if __name__ == '__main__':
    ppgs.train.run(**vars(parse_args()))
