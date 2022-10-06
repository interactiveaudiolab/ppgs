from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'base'


###############################################################################
# Audio parameters
###############################################################################


# Minimum and maximum frequency
FMIN = 50.  # Hz
FMAX = 550.  # Hz

# Audio hopsize
HOPSIZE = 256  # samples

# Maximum sample value of 16-bit audio
MAX_SAMPLE_VALUE = 32768

# Number of spectrogram channels
NUM_FFT = 1024

# Audio sample rate
# SAMPLE_RATE = 22050  # Hz
SAMPLE_RATE = 16000 #TODO check on this

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


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 1000  # steps

# Number of steps between evaluation
EVALUATION_INTERVAL = 2500  # steps


###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 64

# Per-epoch decay rate of the learning rate
LEARNING_RATE_DECAY = .999875

# Number of training steps
NUM_STEPS = 300000

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Partition parameters
###############################################################################

ARCTIC_UNSEEN = ['bdl', 'slt']
TIMIT_UNSEEN = ['ABW0',
    'ADG0',
    'AHH0',
    'AJC0',
    'AKS0',
    'ASW0',
    'AWF0',
    'BDG0',
    'BJK0',
    'BNS0',
    'BPM0',
    'BWM0',
    'CAL1',
    'CAU0',
    'CCS0',
    'CEM0',
    'CFT0',
    'CHH0',
    'CMB0',
    'CMH0',
    'CMH1',
    'CMJ0',
    'CMR0',
    'CRC0',
    'CRH0',
    'CSH0',
    'CTT0',
    'CTW0',
    'DAB0',
    'DAC1',
    'DAC2',
    'DAW1',
    'DBB0',
    'DHC0',
    'DLD0',
    'DLF0',
    'DLS0',
    'DMS0',
    'DRB0',
    'DRD1',
    'DRM0',
    'DRW0',
    'DSC0',
    'DVC0',
    'DWA0',
    'DWK0',
    'EDW0',
    'ELC0',
    'ERS0',
    'ESD0',
    'FGK0',
    'GJD0',
    'GJF0',
    'GLB0',
    'GMD0',
    'GMM0',
    'GRT0',
    'GWR0',
    'GWT0',
    'HES0',
    'HEW0',
    'HPG0',
    'ISB0',
    'JAR0',
    'JAS0',
    'JBR0',
    'JCS0',
    'JDH0',
    'JDM1',
    'JEM0',
    'JES0',
    'JFC0',
    'JJG0',
    'JLM0',
    'JLN0',
    'JMG0',
    'JMP0',
    'JRE0',
    'JRF0',
    'JSA0',
    'JSJ0',
    'JSW0',
    'JTC0',
    'JTH0',
    'JVW0',
    'JWB0',
    'KCH0',
    'KCL0',
    'KDR0',
    'KJL0',
    'KLT0',
    'KMS0',
    'LAS0',
    'LBW0',
    'LIH0',
    'LJB0',
    'LKD0',
    'LLL0',
    'LNH0',
    'LNT0',
    'MAB0',
    'MAF0',
    'MAH0',
    'MCM0',
    'MDB1',
    'MDH0',
    'MDM2',
    'MGD0',
    'MJR0',
    'MLD0',
    'MML0',
    'MWH0',
    'NJM0',
    'NLP0',
    'NLS0',
    'NMR0',
    'PAB0',
    'PAM0',
    'PAM1',
    'PAS0',
    'PCS0',
    'PDF0',
    'PGL0',
    'PKT0',
    'PLB0',
    'PWM0',
    'RAM1',
    'RCS0',
    'RCZ0',
    'REB0',
    'RES0',
    'REW0',
    'RGG0',
    'RJM3',
    'RJM4',
    'RJO0',
    'RJR0',
    'RJS0',
    'RKO0',
    'RMS1',
    'RNG0',
    'ROA0',
    'RPC0',
    'RPP0',
    'RRK0',
    'RTK0',
    'RWS1',
    'SEM0',
    'SFH1',
    'SJS1',
    'SLB0',
    'SLB1',
    'STK0',
    'SXA0',
    'TAA0',
    'TAS1',
    'TDT0',
    'TEB0',
    'THC0',
    'TLH0',
    'TLS0',
    'TMR0',
    'TWH0',
    'UTB0',
    'WBT0',
    'WEW0',
    'WJG0',
    'WVW0',
]


###############################################################################
# Phoneme parameters
###############################################################################

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
    'el': 'l', #TODO check if sufficient
    'em': 'm', #TODO check if sufficient
    'en': 'n', #TODO check if sufficient
    'eng': 'ng', #TODO check if sufficient
    'epi': 'pau', #differs from Kaldi (pau instead of sil)
    'er': 'er',
    'ey': 'ey',
    'f': 'f',
    'g': 'g',
    'gcl': 'bck<g>', #backfill
    'h#': 'pau', #differs from Kaldi (pau instead of sil)
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
    'pau': 'pau', #differs from Kaldi (pau instead of sil)
    'pcl': 'bck<p>', #backfill
    'q': 't', #map to its allophone TODO check the validity of doing this?
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