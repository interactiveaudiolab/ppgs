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
        default=['timit', 'arctic', 'charsiu'],
        choices=['timit', 'arctic', 'charsiu'],
        help="The datasets to purge"
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=ppgs.preprocess.ALL_FEATURES,
        choices=ppgs.preprocess.ALL_FEATURES,
        help="Which cached features to purge. Note that this only affects purges of the cache directory"
    )
    parser.add_argument(
        '--kinds',
        nargs='+',
        default=['cache'],
        choices=['cache', 'datasets', 'sources', 'partitions', 'all'],
        help="Which kinds of local data storage to purge"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Do not prompt user for confirmation"
    )
    return parser.parse_args()


ppgs.data.purge.datasets(**vars(parse_args()))
