from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'base'


###############################################################################
# Audio parameters
###############################################################################


# Audio hopsize
HOPSIZE = 160  # samples

# Number of spectrogram channels
NUM_FFT = 1024

# Number of mel channels
NUM_MELS = 80

# Audio sample rate
SAMPLE_RATE = 16000

# Number of spectrogram channels
WINDOW_SIZE = 1024


###############################################################################
# Data parameters
###############################################################################


# Input and output
ALL_FEATURES = ['audio', 'phonemes']

# All supported input representations
ALL_REPRESENTATIONS = [
    'bottleneck',
    'w2v2fb',
    'w2v2fc',
    'spectrogram',
    'mel',
    'encodec',
    'dac']

# All datasets used by this codebase
DATASETS = ['arctic', 'commonvoice', 'timit']

# Input representation
REPRESENTATION = 'w2v2fb'

# Datasets used for training
TRAINING_DATASET = 'commonvoice'


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of initial downloads before processing into DATA_DIR
SOURCES_DIR = Path(__file__).parent.parent.parent / 'data' / 'sources'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'

# Location of checkpoints
CHECKPOINT_DIR = ASSETS_DIR / 'checkpoints'

# Location of similarity matrix
SIMILARITY_MATRIX_PATH = ASSETS_DIR / 'balanced_similarity.pt'


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of batches to perform during evaluation
EVALUATION_BATCHES = 16

# Number of steps between evaluation
EVALUATION_INTERVAL = 1000  # steps


###############################################################################
# Model parameters
###############################################################################


# Number of attention heads
ATTENTION_HEADS = 2

# Attention window size
ATTENTION_WINDOW_SIZE = 4

# This function takes as input a torch.Device and returns a callable frontend
FRONTEND = None

# Network width
HIDDEN_CHANNELS = 512

# Dimensionality of input representation
INPUT_CHANNELS = 768

# Kernel width
KERNEL_SIZE = 5

# Model architecture.
# One of ['convolution', 'transformer', 'W2V2FC', 'Wav2Vec2.0'].
MODEL = 'transformer'

# Number of hidden layers
NUM_HIDDEN_LAYERS = 5

# Dimensionality of output representation
OUTPUT_CHANNELS = 40


###############################################################################
# Training parameters
###############################################################################


# Number of buckets to partition training examples to minimize padding
BUCKETS = 8

# Whether to use class-balanced loss weights
CLASS_BALANCED = False

# Optimizer step size
LEARNING_RATE = 2e-4

# Maximum number of frames in a batch
MAX_FRAMES = 100000

# Maximum number of frames in a batch during preprocessing
MAX_PREPROCESS_FRAMES = 10000

# Number of training steps
# TEMPORARY
NUM_STEPS = 500000

# Number of data loading worker threads
NUM_WORKERS = 6

# Seed for all random number generators
RANDOM_SEED = 1234
