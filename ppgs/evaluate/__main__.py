from pathlib import Path

import yapecs

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
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint to evaluate')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for evaluation')

    return parser.parse_args()


ppgs.evaluate.datasets(**vars(parse_args()))
