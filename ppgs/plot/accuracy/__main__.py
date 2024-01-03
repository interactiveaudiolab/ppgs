
from pathlib import Path
import yapecs

import ppgs


###############################################################################
# Plot accuracy
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Plot accuracy')
    parser.add_argument(
        '--output_file',
        type=Path,
        help='The location to save the accuracy plot')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=ppgs.DATASETS,
        choices=ppgs.DATASETS,
        help='The datasets to plot')
    parser.add_argument(
        '--representations',
        nargs='+',
        default=ppgs.ALL_REPRESENTATIONS,
        choices=ppgs.ALL_REPRESENTATIONS,
        help='The representations to plot')
    return parser.parse_args()


ppgs.plot.accuracy.from_eval(**vars(parse_args()))
