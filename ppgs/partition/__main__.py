import argparse

import ppgs


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='The datasets to partition')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite existing partitions')
    parser.add_argument(
        '--for-testing',
        action='store_true',
        help='partition datasets entirely for testing (no train or validation sets)'
    )
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    ppgs.partition.datasets(**vars(parse_args()))
