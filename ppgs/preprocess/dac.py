import torch

import ppgs

# Window size of the model
WINDOW_SIZE = 400
HOP_SIZE = 320

###############################################################################
# Preprocess EnCodec input representation
###############################################################################


def from_audios(audio, lengths, sample_rate=ppgs.SAMPLE_RATE, gpu=None):
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    expected_length = audio.shape[-1] // ppgs.HOPSIZE

    # Cache model
    if not hasattr(from_audios, 'model'):
        import dac
        model_path = dac.utils.download(model_type='16khz')
        from_audios.model = dac.DAC.load(model_path)
        from_audios.model = from_audios.model.to(device)

    with torch.autocast(device.type):

        audio = audio.to(device)

        # Encode
        model_input = from_audios.model.preprocess(audio, sample_rate)
        z, codes, latents, _, _ = from_audios.model.encode(model_input)

        # Upsample
        return torch.nn.functional.interpolate(
            codes.to(torch.float),
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
