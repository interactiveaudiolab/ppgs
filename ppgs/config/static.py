"""Config parameters whose values depend on other config parameters"""
import ppgs


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = ppgs.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = ppgs.ASSETS_DIR / 'checkpoints' / 'default.pt'

# weighting file for class balancing
CLASS_WEIGHT_FILE = ppgs.ASSETS_DIR / 'class_weights.pt'


###############################################################################
# Data parameters
###############################################################################


# All possible features to load
ALL_FEATURES = [ppgs.REPRESENTATION, 'phonemes', 'length', 'stem']
