import yapecs

import ppgs


###############################################################################
# Purge datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Measure dataset disk usage')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to measure')
    parser.add_argument(
        '--features',
        nargs='+',
        default=ppgs.preprocess.ALL_FEATURES,
        choices=ppgs.preprocess.ALL_FEATURES,
        help='Which cached features to measure')
    return parser.parse_args()


ppgs.data.measure.datasets(**vars(parse_args()))
