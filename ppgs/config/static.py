"""Config parameters whose values depend on other config parameters"""


import ppgs
from ppgs.preprocess import *


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = ppgs.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = ppgs.ASSETS_DIR / 'checkpoints' / 'default.pt'

# Default configuration file
DEFAULT_CONFIGURATION = ppgs.ASSETS_DIR / 'configs' / 'ppgs.py'


###############################################################################
# Representation
###############################################################################
REPRESENTATION_MAP = {
    'senone': ppgs.preprocess.senone,
    'w2v2fs': ppgs.preprocess.w2v2fs,
    'w2v2fb': ppgs.preprocess.w2v2fb,
    'spectrogram': ppgs.preprocess.spectrogram,
    'mel': ppgs.preprocess.spectrogram #TODO uh oh, need to differentiate
}
