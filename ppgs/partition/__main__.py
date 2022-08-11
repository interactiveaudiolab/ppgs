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
    return parser.parse_args()


if __name__ == '__main__':
    ppgs.partition.datasets(**vars(parse_args()))
