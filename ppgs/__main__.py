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
        '--audio_files',
        type=Path,
        nargs='+',
        help='The speech recordings to compute PPGs for')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        help='The files to save PPGs. Default is audio path with \".pt\" extension.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ppgs.DEFAULT_CHECKPOINT,
        help='The files to save PPGs')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_args()


if __name__ == '__main__':
    ppgs.from_files_to_files(**vars(parse_args()))
