CONFIG = 'charsiutransformerW2V2'
MODULE = 'ppgs'

INPUT_CHANNELS = 768 #dimensionality of wav2vec2 latents
REPRESENTATION = 'w2v2'
MODEL = 'transformer'
NUM_WORKERS=2
EVALUATION_BATCHES = 16
NUM_STEPS = 2000000