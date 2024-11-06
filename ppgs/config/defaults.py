import os
from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'ppgs'


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
ALL_REPRESENTATIONS = ['bottleneck', 'w2v2fb', 'w2v2fc', 'mel', 'encodec']

# All datasets used by this codebase
DATASETS = ['commonvoice', 'arctic', 'timit']

# Best representation
BEST_REPRESENTATION = 'mel'

# Default representation
REPRESENTATION = BEST_REPRESENTATION

# representation kind
# One of ['ppg', 'latents'].
REPRESENTATION_KIND = 'ppg'

# Datasets used for training
TRAINING_DATASET = 'commonvoice'


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
ROOT_DIR = Path(__file__).parent.parent.parent

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of initial downloads before processing into DATA_DIR
SOURCES_DIR = Path(__file__).parent.parent.parent / 'data' / 'sources'

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

# Location of initial downloads before processing into DATA_DIR
SOURCES_DIR = ROOT_DIR / 'data' / 'sources'

# Location of similarity matrix
SIMILARITY_MATRIX_PATH = ASSETS_DIR / 'balanced_similarity.pt'

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
DEFAULT_EVALUATION_STEPS = 16

# Number of steps between evaluation
EVALUATION_INTERVAL = 1000  # steps


###############################################################################
# Model parameters
###############################################################################


# Local checkpoint to use
# If None, Huggingface will be used unless a checkpoint is given in the CLI
LOCAL_CHECKPOINT = None

# Number of attention heads
ATTENTION_HEADS = 2

# Attention window size
ATTENTION_WINDOW_SIZE = 4

# Use causal masking/methods
IS_CAUSAL = False

# This function takes as input a torch.Device and returns a callable frontend
FRONTEND = None

# Network width
HIDDEN_CHANNELS = 256

# Dimensionality of input representation
INPUT_CHANNELS = 80

# Kernel width
KERNEL_SIZE = 5

# Model architecture.
# One of ['convolution', 'transformer', 'W2V2FC', 'Wav2Vec2.0'].
MODEL = 'transformer'

# Number of hidden layers
NUM_HIDDEN_LAYERS = 5

# Dimensionality of output representation
OUTPUT_CHANNELS = 40

# Additional context overlap between chunks
CHUNK_OVERLAP = 50

# Maximum number of frames in a chunk
CHUNK_LENGTH = 500


###############################################################################
# Training parameters
###############################################################################


# Number of buckets to partition training examples to minimize padding
BUCKETS = 1

# Whether to use class-balanced loss weights
CLASS_BALANCED = False

# Infinity norm gradient clipping threshold
GRADIENT_CLIP_THRESHOLD_INF = None

# L2 norm gradient clipping threshold
GRADIENT_CLIP_THRESHOLD_L2 = None

# Optimizer step size
LEARNING_RATE = 2e-4

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 150000

# Maximum number of frames in a batch during preprocessing
MAX_PREPROCESS_FRAMES = 10000

# Number of training steps
STEPS = 500000

# Number of data loading worker threads
# TEMPORARY
# try:
#     NUM_WORKERS = int(os.cpu_count() / max(1, len(GPUtil.getGPUs())))
# except ValueError:
#     NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 8

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Distance parameters
###############################################################################


# A hyperparameter that weights the relative contribution of the acoustic
# phoneme similarity matrix. The default value was tuned to maximize
# correlation between word error rate (WER) and the average JS divergence
# between PPGs.
SIMILARITY_EXPONENT = 1.2
