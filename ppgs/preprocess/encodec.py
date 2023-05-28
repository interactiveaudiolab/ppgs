from encodec import EncodecModel
import torch
from torchaudio.transforms import Resample
import ppgs


def from_audios(
    audio: torch.Tensor,
    lengths,
    gpu=None
):
    expected_length = audio.shape[-1] // ppgs.HOPSIZE
    if not hasattr(from_audios, 'resampler'):
        from_audios.resampler = Resample(orig_freq=16000, new_freq=24000)
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