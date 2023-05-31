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
        '--dataset',
        nargs='+',
        type=Path,
        help='the datasets to process')
    parser.add_argument(
        '--from-features',
        nargs='+',
        help='features to synthesize PPGS from'
    )
    parser.add_argument(
        '--save-from-features',
        action='store_true'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=ppgs.CACHE_DIR,
        help='path to a cache dir different from ppgs.CACHE_DIR, used as input and output'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='The number of worker threads to use for loading data'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    ppgs.process(**vars(parse_args()))
