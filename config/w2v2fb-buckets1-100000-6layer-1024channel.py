MODULE = 'ppgs'

# Configuration name
CONFIG = 'w2v2fb-buckets1-100000-6layer-1024channel'

# Number of buckets to partition training examples to minimize padding
BUCKETS = 1

# Network width
HIDDEN_CHANNELS = 1024

# Dimensionality of input representation
INPUT_CHANNELS = 768

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 100000

# Number of hidden layers
NUM_HIDDEN_LAYERS = 6

# Input representation
REPRESENTATION = 'w2v2fb'
