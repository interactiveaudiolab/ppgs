from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.utils import logging
import torch
import torchaudio
import tqdm
from ppgs.model.transformer import mask_from_lengths

import ppgs

###############################################################################
# Constants
###############################################################################

# W2V2 FB pretrained model config name
W2V2FB_CONFIG = "facebook/wav2vec2-base"

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400
HOP_SIZE = 320


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
            from_features.model.load_state_dict(torch.load(ppgs.CHECKPOINT_DIR / 'w2v2fb.pt')['model'])
        from_features.model = from_features.model.to(features.device)
    return from_features.model(features, new_lengths)

def from_audios(
    audio,
    lengths,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2FB latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=W2V2FB_CONFIG
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Cache model
    if not hasattr(from_audios, 'model'):
        from_audios.model = Wav2Vec2Model.from_pretrained(config).to(device)

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze(dim=1)
    # upsampled_audio = torch.nn.functional.upsample(
    #     audio.reshape(1, 1, len(audio)),
    #     scale_factor=2,
    #     mode='linear'
    # ).squeeze()
    pad = WINDOW_SIZE//2 - HOP_SIZE//2
    padded_audio = torch.nn.functional.pad(audio, (pad, pad))
    # Setup features
    # inputs = from_audio.processor(list(padded_audio), sampling_rate=sample_rate, return_tensors='pt')
    # # interpolated_shape = [inputs.shape[0], inputs.shape[1] * 2]

    # Infer W2V2FB latents
    mask = mask_from_lengths(lengths).squeeze(dim=1).to(torch.long)
    assert len(mask.shape) == 2
    #TODO do we need to squeeze this?
    output = from_audios.model(padded_audio, mask).last_hidden_state
    output = torch.transpose(output, 1, 2)
    upsampled_outputs = torch.nn.functional.interpolate(
        output,
        size=audio.shape[-1]//ppgs.HOPSIZE,
        mode='nearest'
    )
    try:
        assert upsampled_outputs.shape[-1] == audio.shape[-1] // ppgs.HOPSIZE #check that frames are centered and lengths are correct
    except AssertionError:
        import pdb; pdb.set_trace()
    return upsampled_outputs.to(torch.float16)

def from_audio(
    audio,
    sample_rate=None,
    config=None,
    gpu=None):
    """Compute W2V2FB latents from audio"""
    if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
    if config is None: config=W2V2FB_CONFIG
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')


    # Cache model
    if not hasattr(from_audio, 'model'):
        from_audio.model = Wav2Vec2Model.from_pretrained(config).to(device)
    if not hasattr(from_audio, 'processor'):
        from_audio.processor = Wav2Vec2FeatureExtractor.from_pretrained(config)

    # Maybe resample
    audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze()
    # upsampled_audio = torch.nn.functional.upsample(
    #     audio.reshape(1, 1, len(audio)),
    #     scale_factor=2,
    #     mode='linear'
    # ).squeeze()
    pad = WINDOW_SIZE//2 - HOP_SIZE//2
    padded_audio = torch.nn.functional.pad(audio, (pad, pad))

    # Setup features
    inputs = from_audio.processor(padded_audio, sampling_rate=sample_rate, return_tensors='pt')
    # interpolated_shape = [inputs.shape[0], inputs.shape[1] * 2]

    inputs = inputs['input_values'].to(device)

    # Infer W2V2FB latents
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        output = from_audio.model(inputs).last_hidden_state.squeeze().T.unsqueeze(0)
        upsampled_outputs = torch.nn.functional.interpolate(
            output,
            size=audio.shape[-1]//ppgs.HOPSIZE,
            mode='nearest'
        )
        try:
            assert upsampled_outputs.shape[-1] == audio.shape[-1] // ppgs.HOPSIZE #check that frames are centered and lengths are correct
        except AssertionError:
            import pdb; pdb.set_trace()
        return upsampled_outputs


def from_file(audio_file, gpu=None):
    """Compute W2V2FB latents from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute W2V2FB latents from audio file and save to disk"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute W2V2FB latents from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting W2V2FB latents',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)
