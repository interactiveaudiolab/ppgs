import argparse

import ppgs


###############################################################################
# Validate datasets
###############################################################################

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Validate datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['timit', 'arctic', 'charsiu'],
        help='The datasets to purge'
    )
    parser.add_argument(
        '--representation',
        nargs=1,
        default=None,
        help="Which representation to validate"
    )
    parser.add_argument(
        '--partitions',
        nargs='+',
        default=['train', 'valid', 'test'],
        help="Which partitions to validate"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='automatically enter interactive debugger on error'
    )
    return parser.parse_args()


ppgs.data.validate.datasets(**vars(parse_args()))
