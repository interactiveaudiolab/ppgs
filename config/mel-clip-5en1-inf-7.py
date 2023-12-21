MODULE = 'ppgs'

# Configuration name
CONFIG = 'mel-clip-5en1-inf-7'

# Dimensionality of input representation
INPUT_CHANNELS = 80

# Input representation
REPRESENTATION = 'mel'

NUM_HIDDEN_LAYERS = 7
MAX_TRAINING_FRAMES = 120_000

# Clipping parameters
CLIPPING_NORM_TYPE = 'inf'
import torch
CLIPPING_THRESHOLD = 0.5
USE_AUTOCLIP = False
