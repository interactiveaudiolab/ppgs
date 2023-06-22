CONFIG = 'transformerW2V2FS'
MODULE = 'ppgs'

INPUT_CHANNELS = 768 #dimensionality of wav2vec2 latents
REPRESENTATION = 'w2v2fs'
MODEL = 'transformer'
NUM_WORKERS=6
EVALUATION_BATCHES = 16

BATCH_SIZE = 512