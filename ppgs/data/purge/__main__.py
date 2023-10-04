import argparse

import ppgs


###############################################################################
# Purge datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Purge datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to purge')
    parser.add_argument(
        '--features',
        nargs='+',
        default=ppgs.ALL_FEATURES + [rep + '-ppg' for rep in ppgs.ALL_REPRESENTATIONS],
        choices=ppgs.ALL_FEATURES + [rep + '-ppg' for rep in ppgs.ALL_REPRESENTATIONS],
        help='Which features to purge from the cache directory')
    parser.add_argument(
        '--kinds',
        nargs='+',
        default=['cache', 'datasets', 'sources', 'partitions'],
        choices=['cache', 'datasets', 'sources', 'partitions'],
        help='Which kinds of local data storage to purge')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Do not prompt user for confirmation')
    return parser.parse_args()


ppgs.data.purge.datasets(**vars(parse_args()))
