MODULE = 'ppgs'

# Configuration name
CONFIG = 'causal_transformer'

# Network width
HIDDEN_CHANNELS = 256

# Dimensionality of input representation
INPUT_CHANNELS = 80

# Input representation
REPRESENTATION = 'mel'

# Number of training steps
STEPS = 100_000

IS_CAUSAL = True

EVALUATION_INTERVAL = 250
CHECKPOINT_INTERVAL = 2500  # steps
