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
        default=['timit', 'arctic', 'charsiu'],
        help='The datasets to download'
    )
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Skip download step and format only'
    )
    parser.add_argument(
        '--purge-sources',
        action='store_true',
        help='automatically remove original download of dataset upon completion of formatting (free up space)'
    )
    return parser.parse_args()


ppgs.data.download.datasets(**vars(parse_args()))
