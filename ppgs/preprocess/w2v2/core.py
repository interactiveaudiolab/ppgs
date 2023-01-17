from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.utils import logging
import torch
import torchaudio
import tqdm

import ppgs

###############################################################################
# Constants
###############################################################################

# W2V2 pretrained model config name
# W2V2_CONFIG = "facebook/wav2vec2-base-960h"
W2V2_CONFIG = "charsiu/en_w2v2_fs_10ms"

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400


###############################################################################
# Phonetic posteriorgram
###############################################################################

logging.set_verbosity_error()

def from_audio(
    audio,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2 latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=W2V2_CONFIG
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Cache model
    if not hasattr(from_audio, 'model'):
        from_audio.model = Wav2Vec2Model.from_pretrained(config).to(device)
    # if not hasattr(from_audio, 'processor'):
    #     from_audio.processor = Wav2Vec2FeatureExtractor.from_pretrained(config)

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze()

    # Setup features
    # inputs = from_audio.processor(audio, sampling_rate=sample_rate, return_tensors='pt')
    pad = WINDOW_SIZE//2 - ppgs.HOPSIZE//2
    #TODO investigate +1 here
    inputs = torch.nn.functional.pad(audio, (pad, pad+1)).unsqueeze(dim=0)
    # inputs = audio.unsqueeze(dim=0)
    inputs = inputs.to(device)

    # Infer W2V2 latents
    with torch.no_grad():
        output = from_audio.model(inputs).last_hidden_state.squeeze().T
        try:
            assert output.shape[-1] == audio.shape[-1] // ppgs.HOPSIZE #check that frames are centered and lengths are correct
        except AssertionError:
            import pdb; pdb.set_trace()
        return output


def from_file(audio_file, gpu=None):
    """Compute W2V2 latents from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute W2V2 latents from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute W2V2 latents from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting W2V2 latents',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)