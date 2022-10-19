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
HOPSIZE = 160 # samples

# Maximum sample value of 16-bit audio
MAX_SAMPLE_VALUE = 32768

# Number of spectrogram channels
NUM_FFT = 1024

# Audio sample rate
# SAMPLE_RATE = 22050  # Hz
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


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 1000  # steps

# Number of batches to perform during evaluation
EVALUATION_BATCHES = 4

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
# Partition parameters #TODO extract to separate config file
###############################################################################

ARCTIC_UNSEEN = ['bdl', 'slt']
TIMIT_UNSEEN = [
	'MABW0',
	'FADG0',
	'MAHH0',
	'MAJC0',
	'FAKS0',
	'FASW0',
	'FAWF0',
	'MBDG0',
	'MBJK0',
	'MBNS0',
	'MBPM0',
	'MBWM0',
	'FCAL1',
	'FCAU0',
	'MCCS0',
	'MCEM0',
	'FCFT0',
	'MCHH0',
	'MCMB0',
	'FCMH0',
	'FCMH1',
	'MCMJ0',
	'FCMR0',
	'MCRC0',
	'FCRH0',
	'MCSH0',
	'MCTT0',
	'MCTW0',
	'MDAB0',
	'FDAC1',
	'MDAC2',
	'MDAW1',
	'MDBB0',
	'FDHC0',
	'MDLD0',
	'MDLF0',
	'MDLS0',
	'FDMS0',
	'MDRB0',
	'FDRD1',
	'MDRM0',
	'FDRW0',
	'MDSC0',
	'MDVC0',
	'MDWA0',
	'MDWK0',
	'FEDW0',
	'FELC0',
	'MERS0',
	'MESD0',
	'MFGK0',
	'FGJD0',
	'MGJF0',
	'MGLB0',
	'FGMD0',
	'MGMM0',
	'MGRT0',
	'FGWR0',
	'MGWT0',
	'FHES0',
	'FHEW0',
	'MHPG0',
	'FISB0',
	'MJAR0',
	'FJAS0',
	'MJBR0',
	'FJCS0',
	'MJDH0',
	'MJDM1',
	'FJEM0',
	'MJES0',
	'MJFC0',
	'MJJG0',
	'FJLM0',
	'MJLN0',
	'FJMG0',
	'MJMP0',
	'FJRE0',
	'MJRF0',
	'FJSA0',
	'FJSJ0',
	'MJSW0',
	'MJTC0',
	'MJTH0',
	'MJVW0',
	'FJWB0',
	'MKCH0',
	'MKCL0',
	'MKDR0',
	'MKJL0',
	'MKLT0',
	'FKMS0',
	'FLAS0',
	'FLBW0',
	'MLIH0',
	'MLJB0',
	'FLKD0',
	'MLLL0',
	'FLNH0',
	'MLNT0',
	'MMAB0',
	'FMAF0',
	'FMAH0',
	'FMCM0',
	'MMDB1',
	'MMDH0',
	'MMDM2',
	'FMGD0',
	'MMJR0',
	'FMLD0',
	'FMML0',
	'MMWH0',
	'MNJM0',
	'FNLP0',
	'MNLS0',
	'FNMR0',
	'MPAB0',
	'MPAM0',
	'MPAM1',
	'FPAS0',
	'MPCS0',
	'MPDF0',
	'MPGL0',
	'FPKT0',
	'MPLB0',
	'MPWM0',
	'FRAM1',
	'MRCS0',
	'MRCZ0',
	'MREB0',
	'MRES0',
	'FREW0',
	'MRGG0',
	'MRJM3',
	'MRJM4',
	'MRJO0',
	'MRJR0',
	'MRJS0',
	'MRKO0',
	'MRMS1',
	'FRNG0',
	'MROA0',
	'MRPC0',
	'MRPP0',
	'MRRK0',
	'MRTK0',
	'MRWS1',
	'FSEM0',
	'MSFH1',
	'MSJS1',
	'MSLB0',
	'FSLB1',
	'MSTK0',
	'FSXA0',
	'MTAA0',
	'MTAS1',
	'MTDT0',
	'MTEB0',
	'MTHC0',
	'FTLH0',
	'MTLS0',
	'MTMR0',
	'MTWH0',
	'FUTB0',
	'MWBT0',
	'MWEW0',
	'MWJG0',
	'MWVW0',
]


###############################################################################
# Phoneme parameters
###############################################################################

PHONEME_LIST = [
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
	'ax',
	'sp',
	'<unk>'
]

PHONEME_TO_INDEX_MAPPING = {phone: i for i, phone in enumerate(PHONEME_LIST)}

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