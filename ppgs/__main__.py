import argparse
from pathlib import Path

import ppgs


###############################################################################
# Compute phonetic posteriorgram
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Compute phonetic posteriorgram (PPG) features')
    parser.add_argument(
        '--sources',
        nargs='+',
        type=Path,
        help='a list of files and/or directories to process'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='the directory to write features to'
    )
    parser.add_argument(
        '--from-feature',
        default=ppgs.REPRESENTATION,
        help='feature to synthesize PPGS from'
    )
    parser.add_argument(
        '--save-intermediate-features',
        action='store_true',
        help="TODO"
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='The number of worker threads to use for loading data'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    ppgs.process(**vars(parse_args()))
