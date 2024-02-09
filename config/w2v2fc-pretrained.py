MODULE = 'ppgs'

# Configuration name
CONFIG = 'w2v2fc-pretrained'

# Input representation
REPRESENTATION = 'wav'

# Model architecture.
# One of ['convolution', 'transformer', 'W2V2FC', 'Wav2Vec2.0'].
MODEL = 'W2V2FC'

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 1000

# Number of training steps
STEPS = 200000
