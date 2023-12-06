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
        '--audio_paths',
        nargs='+',
        type=Path,
        required=False,
        help='Paths to audio files and/or directories')
    parser.add_argument(
        '--audio_extensions',
        nargs='+',
        default=['wav', 'mp3'],
        help='Extensions for audio files in provided directories'
    )
    parser.add_argument(
        '--ppg_paths',
        nargs='+',
        type=Path,
        required=False,
        help='Paths to PPG files and/or directories')
    parser.add_argument(
        '--textgrid_paths',
        nargs='+',
        type=Path,
        required=False,
        help='Paths to textgrid files and/or directories')
    parser.add_argument(
        '--output_paths',
        type=Path,
        nargs='+',
        default=None,
        help='The one-to-one corresponding output paths')
    parser.add_argument(
        '--video',
        action='store_true',
        help='Create video visualizations instead of images'
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Save resulting images as pdf files'
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='The checkpoint file')
    parser.add_argument(
        '--font_filename',
        default=None,
        help='The font file to use for text'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_args()


ppgs.plot.from_paths_to_paths(**vars(parse_args()))
