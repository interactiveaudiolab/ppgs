import argparse

import ppgs


###############################################################################
# Purge datasets
###############################################################################

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Measure dataset disk usage')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['timit', 'arctic', 'charsiu'],
        choices=['timit', 'arctic', 'charsiu'],
        help="The datasets to measure"
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=ppgs.preprocess.ALL_FEATURES,
        choices=ppgs.preprocess.ALL_FEATURES,
        help="Which cached features to measure"
    )
    parser.add_argument(
        '--unit',
        default='B',
        choices=['B', 'KB', 'MB', 'GB', 'TB'],
        help='Unit to print filesizes with'
    )
    return parser.parse_args()


ppgs.data.measure.datasets(**vars(parse_args()))
