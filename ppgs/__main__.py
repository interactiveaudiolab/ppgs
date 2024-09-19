import yapecs
from pathlib import Path

import ppgs


###############################################################################
# PPG inference command-line interface
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Phonetic posteriorgram inference')
    parser.add_argument(
        '--audio_files',
        nargs='+',
        type=Path,
        required=True,
        help='Paths to audio files')
    parser.add_argument(
        '--output_files',
        type=Path,
        required=True,
        nargs='+',
        help='The one-to-one corresponding output files')
    parser.add_argument(
        '--representation',
        type=str,
        default=ppgs.REPRESENTATION,
        help='Representation to use for inference')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of CPU threads for multiprocessing')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=ppgs.MAX_INFERENCE_FRAMES,
        help='Maximum number of frames in a batch')
    parser.add_argument(
        '--legacy-mode',
        action='store_true',
        help='Use legacy (unchunked) inference'
    )
    return parser.parse_args()


ppgs.from_files_to_files(**vars(parse_args()))
