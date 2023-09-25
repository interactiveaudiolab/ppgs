MODULE = 'ppgs'

# Configuration name
CONFIG = 'w2v2ft'

# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 2000  # steps

# Number of steps between evaluation
EVALUATION_INTERVAL = 500  # steps

# Kernel width
KERNEL_SIZE = 1

# Optimizer step size
LEARNING_RATE = 1e-6

# Maximum number of frames in a batch
MAX_FRAMES = 70000

# Model architecture.
# One of ['convolution', 'transformer', 'W2V2FC', 'Wav2Vec2.0'].
MODEL = 'Wav2Vec2.0'

# Number of training steps
NUM_STEPS = 50000

# Input representation
REPRESENTATION = 'wav'
