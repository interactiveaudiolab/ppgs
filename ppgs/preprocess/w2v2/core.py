from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import torchaudio
import tqdm

import ppgs

###############################################################################
# Constants
###############################################################################

#TODO reconfigure these
# # PPG model checkpoint file
# CHECKPOINT_FILE = ppgs.ASSETS_DIR / 'checkpoints' / 'ppg.pt'

# # PPG model configuration
# CONFIG_FILE = ppgs.ASSETS_DIR / 'configs' / 'ppg.yaml'

# W2V2 pretrained model config name
W2V2_CONFIG = "facebook/wav2vec2-base-960h"

# Sample rate of the PPG model
SAMPLE_RATE = 16000


###############################################################################
# Phonetic posteriorgram
###############################################################################


def from_audio(
    audio,
    sample_rate=ppgs.SAMPLE_RATE,
    config=W2V2_CONFIG,
    gpu=None):
    """Compute PPGs from audio"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Cache model
    if not hasattr(from_audio, 'model'):
        from_audio.model = Wav2Vec2Model.from_pretrained(config).to(device)
    if not hasattr(from_audio, 'processor'):
        from_audio.processor = Wav2Vec2FeatureExtractor.from_pretrained(config)

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze()

    # Setup features
    inputs = from_audio.processor(audio, sampling_rate=sample_rate, return_tensors='pt')
    inputs = inputs.to(device)
    # length = torch.tensor([audio.shape[-1]], dtype=torch.long, device=device)

    # Infer ppgs
    with torch.no_grad():
        # return from_audio.model(audio, length)[0].T
        return from_audio.model(**inputs).last_hidden_state


def from_file(audio_file, gpu=None):
    """Compute PPGs from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute PPGs from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute PPGs from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting W2V2 latents',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)