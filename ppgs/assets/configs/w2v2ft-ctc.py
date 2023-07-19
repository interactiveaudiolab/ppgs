CONFIG = 'w2v2ft-ctc'
MODULE = 'ppgs'

REPRESENTATION = 'w2v2ft'
MODEL = 'Wav2Vec2.0'
NUM_WORKERS=4
EVALUATION_BATCHES=16
MAX_FRAMES=70000
NUM_STEPS=20000
EVALUATION_INTERVAL = 100
CHECKPOINT_INTERVAL = 2000

KERNEL_SIZE = 9

GRAD_2_CLIP = 0.25
GRAD_INF_CLIP = 0.1

LEARNING_RATE = 1e-6

LOSS_FUNCTION = 'CTC'
OUTPUT_CHANNELS = 41

import torch
def _backend(predicted_logits: torch.Tensor):
    """predicted_logits are BATCH x DIMS x TIME"""
    for sequence_logits in predicted_logits:
        current_prediction = None
        for timestep in sequence_logits.T:
            if current_prediction is None: # first timestep
                current_prediction = timestep.argmax()
            elif timestep.argmax() == len(timestep) - 1: # prediction is BLANK
                # assign probability to current prediction instead
                timestep[current_prediction] = timestep[-1] #TODO try summing instead?
            else:
                current_prediction = timestep.argmax()
    return predicted_logits[:, :-1, :]

BACKEND = _backend