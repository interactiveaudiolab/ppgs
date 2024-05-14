import torch
import torchutil
import transformers

import ppgs

# Turn off logging
transformers.utils.logging.set_verbosity_error()


###############################################################################
# Constants
###############################################################################


# W2V2 FB pretrained model config name
W2V2FB_CONFIG = 'facebook/wav2vec2-base'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

# Window size of the model
WINDOW_SIZE = 400
HOP_SIZE = 320


###############################################################################
# Preprocess wav2vec 2.0 latents
###############################################################################


def from_audios(
    audio,
    lengths,
    sample_rate=None,
    gpu=None):
    """Compute W2V2FB latents from audio"""
    with torch.no_grad():
        if sample_rate is None: sample_rate=ppgs.SAMPLE_RATE
        config=W2V2FB_CONFIG
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audios, 'model') or from_audios.device != device:
            from_audios.model = transformers.Wav2Vec2Model.from_pretrained(
                config).to(device)
            from_audios.device = device

        # Maybe resample
        audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).to(device)

        # Pad
        lengths = torch.ceil(
            lengths * (SAMPLE_RATE / sample_rate)
        ).to(torch.long)
        pad = WINDOW_SIZE // 2 - HOP_SIZE // 2
        padded_audio = torch.nn.functional.pad(
            audio,
            (pad, pad)
        ).squeeze(dim=1)

        # Infer W2V2FB latents
        mask = ppgs.model.transformer.mask_from_lengths(
            lengths, pad
        ).squeeze(dim=1).to(torch.long).to(audio.device)
        output = from_audios.model(padded_audio, mask).last_hidden_state
        output = torch.transpose(output, 1, 2)

        # Upsample
        upsampled_outputs = torch.nn.functional.interpolate(
            output,
            size=audio.shape[-1] // ppgs.HOPSIZE,
            mode='nearest')

        return upsampled_outputs.to(torch.float16)

def from_audio(
    audio: torch.Tensor,
    sample_rate: float = None,
    gpu=None):
    """Compute W2V2FB latents from audio"""

    with torch.no_grad():
        num_dims = audio.dim()
        if num_dims == 1:
            audio = audio.unsqueeze(dim=0)
        predicted_ppgs = from_audios(
            audio=audio,
            lengths = torch.tensor([audio.shape[-1]]),
            sample_rate=sample_rate,
            gpu=gpu)
        if num_dims == 1:
            predicted_ppgs = predicted_ppgs.squeeze(dim=0)
        return predicted_ppgs


def from_file(audio_file, gpu=None):
    """Compute W2V2FB latents from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute W2V2FB latents from audio file and save to disk"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute W2V2FB latents from audio files and save to disk"""
    for audio_file, output_file in torchutil.iterator(
        zip(audio_files, output_files),
        'Extracting W2V2FB latents',
        total=len(audio_files)
    ):
        from_file_to_file(audio_file, output_file, gpu)
