MODULE = 'ppgs'

# Configuration name
CONFIG = 'mamba-mel'

# Model architecture
MODEL = 'mamba'

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 100000

CHECKPOINT_INTERVAL = 1000  # steps

INPUT_CHANNELS = 80

# Network width
HIDDEN_CHANNELS = 1024

# Dimensionality of input representation

# Number of hidden layers
NUM_HIDDEN_LAYERS = 16

LEARNING_RATE = 5e-7

GRADIENT_CLIP_THRESHOLD_INF = 1.0

# Input representation
REPRESENTATION = 'mel'

# Number of training steps
STEPS = 25_000
