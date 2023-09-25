from pathlib import Path
import pypar


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'base'


###############################################################################
# Audio parameters
###############################################################################


# All supported input representations
ALL_REPRESENTATIONS = [
    'bottleneck',
    'w2v2fb',
    'w2v2fc',
    'spectrogram',
    'mel',
    'encodec',
    'w2v2ft']

# Audio hopsize
HOPSIZE = 160  # samples

# Number of spectrogram channels
NUM_FFT = 1024

# Number of mel channels
NUM_MELS = 80

# Input representation
REPRESENTATION = 'w2v2fb'

# Audio sample rate
SAMPLE_RATE = 16000

# Number of spectrogram channels
WINDOW_SIZE = 1024


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


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of batches to perform during evaluation
EVALUATION_BATCHES = 16

# Number of steps between evaluation
EVALUATION_INTERVAL = 1000  # steps

# Apprise services to send job notifications
NOTIFICATION_SERVICES = []


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

# Number of training steps
NUM_STEPS = 200000

# Number of data loading worker threads
NUM_WORKERS = 6

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Phoneme parameters
###############################################################################


# TODO - move to new file
PHONEMES = [
	'aa',
	'ae',
	'ah',
	'ao',
	'aw',
	'ay',
	'b',
	'ch',
	'd',
	'dh',
	'eh',
	'er',
	'ey',
	'f',
	'g',
	'hh',
	'ih',
	'iy',
	'jh',
	'k',
	'l',
	'm',
	'n',
	'ng',
	'ow',
	'oy',
	'p',
	'r',
	's',
	'sh',
	't',
	'th',
	'uh',
	'uw',
	'v',
	'w',
	'y',
	'z',
	'zh',
	pypar.SILENCE]

CHARSIU_PHONE_ORDER = [
    pypar.SILENCE,
    'ng',
    'f',
    'm',
    'ae',
    'r',
    'uw',
    'n',
    'iy',
    'aw',
    'v',
    'uh',
    'ow',
    'aa',
    'er',
    'hh',
    'z',
    'k',
    'ch',
    'w',
    'ey',
    'zh',
    't',
    'eh',
    'y',
    'ah',
    'b',
    'p',
    'th',
    'dh',
    'ao',
    'g',
    'l',
    'jh',
    'oy',
    'sh',
    'd',
    'ay',
    's',
    'ih']

CHARSIU_PERMUTE = [CHARSIU_PHONE_ORDER.index(phone) for phone in PHONEMES]

PHONEME_TO_INDEX_MAPPING = {phone: i for i, phone in enumerate(PHONEMES)}

TIMIT_TO_ARCTIC_MAPPING = {
    'aa': 'aa',
    'ae': 'ae',
    'ah': 'ah',
    'ao': 'ao', #differs from Kaldi, likely an error in Kaldi
    'aw': 'aw',
    'ax': 'ah',
    'ax-h': 'ah',
    'axr': 'er',
    'ay': 'ay',
    'b': 'b',
    'bcl': 'bck<b>', #backfill
    'ch': 'ch',
    'd': 'd',
    'dcl': 'bck<d,jh>', #backfill
    'dh': 'dh',
    'dx': 'd', #assumption
    'eh': 'eh',
    'el': 'l',
    'em': 'm',
    'en': 'n',
    'eng': 'ng',
    'epi': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'er': 'er',
    'ey': 'ey',
    'f': 'f',
    'g': 'g',
    'gcl': 'bck<g>', #backfill
    'h#': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'hh': 'hh',
    'hv': 'hh',
    'ih': 'ih',
    'ix': 'ih',
    'iy': 'iy',
    'jh': 'jh',
    'k': 'k',
    'kcl': 'bck<k>', #backfill
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ng',
    'nx': 'n',
    'ow': 'ow',
    'oy': 'oy',
    'p': 'p',
    'pau': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'pcl': 'bck<p>', #backfill
    'q': 't', #map to its allophone
    'r': 'r',
    's': 's',
    'sh': 'sh',
    't': 't',
    'tcl': 'bck<t,ch>', #backfill
    'th': 'th',
    'uh': 'uh',
    'uw': 'uw',
    'ux': 'uw',
    'v': 'v',
    'w': 'w',
    'y': 'y',
    'z': 'z',
    'zh': 'zh' #differs from Kaldi
}
