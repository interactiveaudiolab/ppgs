import torch
import torchaudio

import ppgs


###############################################################################
# Preprocess EnCodec input representation
###############################################################################


def from_audios(audio, lengths, sample_rate=ppgs.SAMPLE_RATE, gpu=None):
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    expected_length = audio.shape[-1] // ppgs.HOPSIZE
    # Cache resampler
    if (
        not hasattr(from_audios, 'resampler') or
        sample_rate != from_audios.sample_rate
    ):
        from_audios.sample_rate = sample_rate
        from_audios.resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=24000
        ).to(device)

    # Cache model
    if not hasattr(from_audios, 'model'):
        from encodec import EncodecModel
        from_audios.model = EncodecModel.encodec_model_24khz()
        from_audios.model.to(device)

    with torch.autocast(device.type):

        # Resample to 24khz
        audio = audio.to(device)
        audio = from_audios.resampler(audio)

        # Encode
        output = from_audios.model.encode(audio)[0][0].to(torch.float32)

        # Upsample
        return torch.nn.functional.interpolate(
            output,
            size=expected_length,
            mode='nearest'
        ).to(torch.int)


def from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, gpu=None):
    if audio.dim() == 2:
        audio = audio.unsqueeze(dim=0)
    return from_audios(
        audio,
        audio.shape[-1],
        sample_rate=sample_rate,
        gpu=gpu)
