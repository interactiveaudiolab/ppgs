"""Config parameters whose values depend on other config parameters"""


import ppgs
from ppgs.model import BaselineModel


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
# Model
###############################################################################

MODEL = BaselineModel()