import yapecs
from pathlib import Path

import ppgs


###############################################################################
# PPG inference command-line interface
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Trace ppg inference model to TorchScript module')
    parser.add_argument(
        '--input_file',
        type=Path,
        required=True,
        help='Path to audio file.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    parser.add_argument(
        '--representation',
        type=str,
        default=None,
        help='Representation to use for inference'
    )
    parser.add_argument(
        '--module_file',
        type=Path,
        default=None,
        required=True,
        help='Save parsed TorchScript module'
    )
    return parser.parse_args()


ppgs.trace.trace(**vars(parse_args()))
