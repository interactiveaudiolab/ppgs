"""__main__.py - entry point for ppgs.preprocess"""


import argparse

import ppgs


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['arctic'],
        help='The name of the datasets to use')
    parser.add_argument(
        '--features',
        nargs='+',
        help='The features to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    parser.add_argument(
        '--use-cached-inputs',
        action='store_true',
        help='Use cache dir for inputs (more space efficient)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=-1,
        help='The number of worker threads to use for loading data'
    )
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    ppgs.preprocess.datasets(**vars(parse_args()))
