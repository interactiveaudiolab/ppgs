import argparse

import ppgs


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to partition')
    return parser.parse_known_args()[0]


ppgs.partition.datasets(**vars(parse_args()))
