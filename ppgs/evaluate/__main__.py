"""__main__.py - entry point for ppgs.evaluate"""


import argparse
from pathlib import Path

import ppgs


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['arctic'],
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ppgs.DEFAULT_CHECKPOINT,
        help='The checkpoint file to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')
    parser.add_argument(
        '--partition',
        default='test',
        choices=['train', 'valid', 'test']
    )

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    ppgs.evaluate.datasets(**vars(parse_args()))
