MODULE = 'ppgs'

# Configuration name
CONFIG = 'mamba'

# Model architecture
MODEL = 'mamba'

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 50000

# Network width
HIDDEN_CHANNELS = 1024

# Dimensionality of input representation
INPUT_CHANNELS = 768

# Number of hidden layers
NUM_HIDDEN_LAYERS = 24

LEARNING_RATE = 1e-6

# Input representation
REPRESENTATION = 'w2v2fb'

# Number of training steps
STEPS = 25_000
