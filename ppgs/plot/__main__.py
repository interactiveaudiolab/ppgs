import yapecs
from pathlib import Path

import ppgs


###############################################################################
# PPG inference command-line interface
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Phonetic posteriorgram plotting')
    parser.add_argument(
        '--audio_files',
        nargs='+',
        type=Path,
        help='Audio filenames')
    parser.add_argument(
        '--ppg_files',
        nargs='+',
        type=Path,
        help='PPG filenames')
    parser.add_argument(
        '--second_ppg_files',
        nargs='+',
        type=Path,
        help='Second PPG filenames to compare to')
    parser.add_argument(
        '--textgrid_files',
        nargs='+',
        type=Path,
        help='TextGrid files containing lexical alignments')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        help='The one-to-one corresponding output files')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The checkpoint file')
    parser.add_argument(
        '--font_filename',
        type=Path,
        help='The font file to use for text')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_args()


ppgs.plot.from_files_to_files(**vars(parse_args()))
