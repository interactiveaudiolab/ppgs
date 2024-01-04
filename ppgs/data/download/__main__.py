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
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to download')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Skip download step and format only')
    return parser.parse_args()


ppgs.data.download.datasets(**vars(parse_args()))
