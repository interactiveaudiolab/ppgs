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
        default='timit arctic',
        help='The datasets to download'
    )
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Skip download step and format only'
    )
    parser.add_argument(
        '--timit-source',
        nargs=1,
        default=None,
        type=str,
        help='path to the timit source that should be used'
    )
    parser.add_argument(
        '--arctic-speakers',
        choices=[
            'awb',
            'bdl',
            'clb',
            'jmk',
            'ksp',
            'rms',
            'slt',
        ],
        nargs='*',
        default=[
            'awb',
            'bdl',
            'clb',
            'jmk',
            'ksp',
            'rms',
            'slt',
        ],
        help='specify for which speakers data should be downloaded'
    )
    return parser.parse_args()


ppgs.data.download.datasets(**vars(parse_args()))
