MODULE = 'ppgs'

# Configuration name
CONFIG = 'w2v2fb-buckets1-150000-5layer-512channel-clipinf'

# Number of buckets to partition training examples to minimize padding
BUCKETS = 1

# Method to use for gradient clipping.
# One of ['autoclip', 'inf', 'l1', 'l2', 'skip'].
GRADIENT_CLIPPING_METHOD = 'inf'

# Gradient clipping threshold.
# For autoclip, this specifies the clipping percentile.
GRADIENT_CLIPPING_THRESHOLD = 10000

# Network width
HIDDEN_CHANNELS = 512

# Dimensionality of input representation
INPUT_CHANNELS = 768

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 150000

# Number of hidden layers
NUM_HIDDEN_LAYERS = 5

# Input representation
REPRESENTATION = 'w2v2fb'
