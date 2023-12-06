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
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The name of the datasets to use')
    parser.add_argument(
        '--representations',
        nargs='+',
        default=[ppgs.REPRESENTATION],
        choices=ppgs.ALL_REPRESENTATIONS,
        help='The representations to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=ppgs.NUM_WORKERS,
        help='The number of worker threads to use for loading data')
    parser.add_argument(
        '--partition',
        help='The partition to preprocess. Uses all partitions by default.')
    return parser.parse_args()


ppgs.preprocess.datasets(**vars(parse_args()))
