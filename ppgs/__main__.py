import yapecs
from pathlib import Path

import ppgs


###############################################################################
# Compute phonetic posteriorgram
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Compute phonetic posteriorgram (PPG) features')
    parser.add_argument(
        '--sources',
        nargs='+',
        type=Path,
        help='a list of files and/or directories to process')
    parser.add_argument(
        '--sinks',
        type=Path,
        nargs='+',
        help='the directory to write features to')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='The number of worker threads to use for loading data')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    parser.add_argument(
        '--representation',
        default=ppgs.REPRESENTATION,
        help='feature to synthesize PPGS from')
    parser.add_argument(
        '--save-intermediate-features',
        action='store_true',
        help="save the intermediate features from which PPGs are computed (e.g. w2v2fb)")
    return parser.parse_args()


if __name__ == '__main__':
    ppgs.from_sources_to_sinks(**vars(parse_args()))
