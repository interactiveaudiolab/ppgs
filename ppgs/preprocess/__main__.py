"""__main__.py - entry point for ppgs.preprocess"""


import yapecs

import ppgs


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = yapecs.ArgumentParser(description='Preprocess a dataset')
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
        '--num-workers',
        type=int,
        default=0,
        help='The number of worker threads to use for loading data'
    )
    return parser.parse_args()


if __name__ == '__main__':
    ppgs.preprocess.datasets(**vars(parse_args()))
