import argparse

import ppgs


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Dataset statistics')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to get statistics for')
    return parser.parse_known_args()[0]


ppgs.data.stats.process(**vars(parse_args()))
