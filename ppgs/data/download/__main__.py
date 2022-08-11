import argparse

import ppgs


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to download')
    return parser.parse_args()


ppgs.data.download.datasets(**vars(parse_args()))
