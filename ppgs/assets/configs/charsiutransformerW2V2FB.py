CONFIG = 'charsiutransformerW2V2FB'
MODULE = 'ppgs'

INPUT_CHANNELS = 768 #dimensionality of wav2vec2 latents
REPRESENTATION = 'w2v2fb'
MODEL = 'transformer'
NUM_WORKERS=10
EVALUATION_BATCHES = 16
NUM_STEPS = 1000000
BATCH_SIZE = 512