import torch
import tqdm
from transformers import Wav2Vec2Model
from transformers.utils import logging

import ppgs
from ppgs.model.transformer import mask_from_lengths

###############################################################################
# Constants
###############################################################################

# W2V2 FS pretrained model config name
# W2V2_CONFIG = "facebook/wav2vec2-base-960h"
W2V2FS_CONFIG = "charsiu/en_w2v2_fs_10ms"

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400


###############################################################################
# Phonetic posteriorgram
###############################################################################

logging.set_verbosity_error()

def from_features(
    features: torch.Tensor,
    new_lengths: torch.Tensor,
    checkpoint=None,
    gpu=0
):
    if not hasattr(from_features, 'model'):
        from_features.model = ppgs.Model()()
        if checkpoint is not None:
            from_features.model.load_state_dict(torch.load(checkpoint)['model'])
        else:
            from_features.model.load_state_dict(torch.load(ppgs.CHECKPOINT_DIR / 'w2v2fs.pt')['model'])
        from_features.model.to(features.device)
    return from_features.model(features, new_lengths)

def from_audios(
    audio,
    lengths,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2FS latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=W2V2FS_CONFIG
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
    inputs = torch.nn.functional.pad(audio, (pad, pad+1))
    # inputs = audio.unsqueeze(dim=0)

    # Infer W2V2FS latents
    mask = mask_from_lengths(lengths).squeeze(dim=1).to(torch.long)
    assert len(mask.shape) == 2
    output = from_audio.model(inputs, mask).last_hidden_state.squeeze()
    output = torch.transpose(output, 1, 2)
    try:
        assert output.shape[-1] == audio.shape[-1] // ppgs.HOPSIZE #check that frames are centered and lengths are correct
    except AssertionError:
        import pdb; pdb.set_trace()
    return output.to(torch.float16)

def from_audio(
    audio,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2FS latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=W2V2FS_CONFIG
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

    # Infer W2V2FS latents
    with torch.no_grad():
        output = from_audio.model(inputs).last_hidden_state.squeeze().T
        try:
            assert output.shape[-1] == audio.shape[-1] // ppgs.HOPSIZE #check that frames are centered and lengths are correct
        except AssertionError:
            import pdb; pdb.set_trace()
        return output


def from_file(audio_file, gpu=None):
    """Compute W2V2FS latents from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute W2V2FS latents from audio file and save to disk"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute W2V2FS latents from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting W2V2FS latents',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)
