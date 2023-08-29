import torch
from encodec import EncodecModel
from torchaudio.transforms import Resample

import ppgs


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
            from_features.model.load_state_dict(torch.load(ppgs.CHECKPOINT_DIR / 'encodec.pt')['model'])
        from_features.frontend = ppgs.FRONTEND(features.device)
        from_features.model.to(features.device)
    
    return from_features.model(from_features.frontend(features), new_lengths)

def from_audios(
    audio: torch.Tensor,
    lengths,
    sample_rate=ppgs.SAMPLE_RATE,
    gpu=None
):
    expected_length = audio.shape[-1] // ppgs.HOPSIZE
    if not hasattr(from_audios, 'resampler') or sample_rate != from_audios.sample_rate:
        from_audios.sample_rate = sample_rate
        from_audios.resampler = Resample(orig_freq=sample_rate, new_freq=24000)
        from_audios.resampler.to(audio.device)
    #resample to 24khz
    audio = from_audios.resampler(audio)
    # Cache model
    if not hasattr(from_audios, 'model'):
        from_audios.model = EncodecModel.encodec_model_24khz()
        from_audios.model.to(audio.device)

    output = from_audios.model.encode(audio)[0][0].to(torch.float32)
    upsampled_outputs = torch.nn.functional.interpolate(
        output,
        size=expected_length,
        mode='nearest'
    ).to(torch.int)
    #this messes up padding, but we use lengths to mask anyway
    # return upsampled_outputs
    return upsampled_outputs

def from_audio(
    audio: torch.Tensor,
    sample_rate=ppgs.SAMPLE_RATE,
    gpu=None
):
    if audio.dim() == 2:
        audio = audio.unsqueeze(dim=0)
    return from_audios(
        audio,
        audio.shape[-1],
        sample_rate=sample_rate,
        gpu=gpu
    )
