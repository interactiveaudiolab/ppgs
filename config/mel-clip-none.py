MODULE = 'ppgs'

# Configuration name
CONFIG = 'mel-clip-none'

# Dimensionality of input representation
INPUT_CHANNELS = 80

# Input representation
REPRESENTATION = 'mel'

NUM_HIDDEN_LAYERS = 6
MAX_TRAINING_FRAMES = 130_000

# Clipping parameters
CLIPPING_NORM_TYPE = 1.0
import torch
CLIPPING_THRESHOLD = torch.inf
USE_AUTOCLIP = False
