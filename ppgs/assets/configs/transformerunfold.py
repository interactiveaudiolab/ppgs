CONFIG = 'transformerunfold'
MODULE = 'ppgs'

INPUT_CHANNELS = 1024 #dimensionality of unfold latents
REPRESENTATION = 'unfold'
MODEL = 'transformer'
NUM_WORKERS=10
EVALUATION_BATCHES = 16
NUM_STEPS = 500000
BATCH_SIZE = 512