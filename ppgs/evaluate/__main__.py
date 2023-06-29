"""__main__.py - entry point for ppgs.evaluate"""


import yapecs
from pathlib import Path

import ppgs


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['arctic'],
        help='The datasets to evaluate')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint',
        type=Path,
        default=ppgs.DEFAULT_CHECKPOINT,
        dest='model_source',
        help='The checkpoint file to evaluate')
    group.add_argument(
        '--run',
        type=Path,
        dest='model_source',
        help='The run directory of the model to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')
    parser.add_argument(
        '--partition',
        default='test',
        choices=['train', 'valid', 'test'])

    return parser.parse_args()


if __name__ == '__main__':
    ppgs.evaluate.datasets(**vars(parse_args()))
