CONFIG = 'w2v2fb-ctc'
MODULE = 'ppgs'

INPUT_CHANNELS = 768 #dimensionality of wav2vec2 latents
REPRESENTATION = 'w2v2fb'
MODEL = 'transformer'
NUM_WORKERS=6
EVALUATION_BATCHES = 16

NUM_HIDDEN_LAYERS = 5
MAX_FRAMES = 10000
HIDDEN_CHANNELS = 1024

EVALUATION_INTERVAL = 500

OUTPUT_CHANNELS = 41
LOSS_FUNCTION = 'CTC'

LEARNING_RATE=1e-4

import torch
def _backend_0(predicted_logits: torch.Tensor):
    """predicted_logits are BATCH x DIMS x TIME"""
    for sequence_logits in predicted_logits:
        current_prediction = None
        for timestep in sequence_logits.T:
            timestep_prediction = timestep.argmax()
            if current_prediction is None: # first timestep
                current_prediction = timestep_prediction
            elif timestep_prediction == len(timestep) - 1: # prediction is BLANK
                # assign probability to current prediction instead
                timestep[current_prediction] = timestep[-1] #TODO try summing instead?
            else:
                current_prediction = timestep_prediction
    return predicted_logits[:, :-1, :]

def _backend_1(predicted_logits: torch.Tensor):
    """predicted_logits are BATCH x DIMS x TIME"""
    predictions = predicted_logits.argmax(dim=1)
    blank_indices = torch.argwhere(predictions == 40)
    for batch, time in blank_indices:
        previous_timestep_max = predictions[batch, time-1]
        predictions[batch, time] = predictions[batch, time-1]
        predicted_logits[batch, previous_timestep_max, time] = predicted_logits[batch, 40, time]
    return predicted_logits[:, :-1, :]

def _backend_2(predicted_logits: torch.Tensor):
    """predicted_logits are BATCH x DIMS x TIME"""
    window_radius = (3, 3)
    predictions = predicted_logits.argmax(dim=1)
    predictions[..., -1] = 39
    blank_indices = torch.argwhere(predictions == 40)
    forward_predictions = predictions.clone()
    backward_predictions = predictions.clone()
    max_timesteps = predicted_logits.shape[-1] - 1
    # get forward fill indices
    for batch, time in blank_indices:
        forward_predictions[batch, time] = forward_predictions[batch, max(time-1, 0)]
    # get backward fill indices
    for batch, time in reversed(blank_indices):
        backward_predictions[batch, time] = backward_predictions[batch, min(time+1, max_timesteps)]
    # choose between backward and forward
    for batch, time in blank_indices:
        forward_choice = forward_predictions[batch, time]
        backward_choice = backward_predictions[batch, time]
        choices = torch.tensor([forward_choice, backward_choice])
        window = predicted_logits[batch, choices, max(0, time-window_radius[0]):min(max_timesteps, time+window_radius[1])+1]
        # votes = window.argmax(dim=0)
        vote = window.sum(dim=1).argmax()
        # decision = choices[int(votes.mean(dtype=torch.float).round().item())]
        decision = choices[vote]
        predictions[batch, time] = decision
        predicted_logits[batch, decision, time] = predicted_logits[batch, 40, time]
    return predicted_logits[:, :-1, :]

def _backend_3(predicted_logits: torch.Tensor):
    """predicted_logits are BATCH x DIMS x TIME"""
    predicted_logits = predicted_logits.clone()
    predictions = predicted_logits.argmax(dim=1)
    probabilities = torch.softmax(predicted_logits.to(torch.float), dim=1)
    predictions[..., -1] = 39
    # choose between backward and forward
    for batch_idx in range(0, predicted_logits.shape[0]):
        time = 1
        while time < predicted_logits.shape[-1]-1:
            if predicted_logits[batch_idx, ..., time].argmax() != 40:
                time += 1
            else:
                start = time
                #discover end of chain
                end = (predictions[batch_idx, time:]!=40).to(torch.long).argmax() + start
                start_prediction = predictions[batch_idx, start-1]
                end_prediction = predictions[batch_idx, min(end, predictions.shape[-1])]
                if start_prediction == end_prediction:
                    time = end
                    predicted_logits[batch_idx, start_prediction, start:end] = predicted_logits[batch_idx, 40, start:end]
                else:
                    blank_confidence = probabilities[batch_idx, 40, start:end]
                    boundary = blank_confidence.argmin() + 1 + start
                    predicted_logits[batch_idx, start_prediction, start:boundary] = predicted_logits[batch_idx, 40, start:boundary]
                    predicted_logits[batch_idx, end_prediction, boundary:end] = predicted_logits[batch_idx, 40, boundary:end]
                    time = end
    # for batch, time in blank_indices:
    #     forward_choice = forward_predictions[batch, time]
    #     backward_choice = backward_predictions[batch, time]
    #     choices = torch.tensor([forward_choice, backward_choice])
    #     window = predicted_logits[batch, choices, max(0, time-window_radius[0]):min(max_timesteps, time+window_radius[1])+1]
    #     # votes = window.argmax(dim=0)
    #     vote = window.sum(dim=1).argmax()
    #     # decision = choices[int(votes.mean(dtype=torch.float).round().item())]
    #     decision = choices[vote]
    #     predictions[batch, time] = decision
    #     predicted_logits[batch, decision, time] = predicted_logits[batch, 40, time]
    return predicted_logits[:, :-1, :]

BACKEND = _backend_3