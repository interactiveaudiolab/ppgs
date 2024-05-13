import torch
import torchutil
import transformers

import ppgs

# Turn off logging
transformers.utils.logging.set_verbosity_error()


###############################################################################
# Constants
###############################################################################


# W2V2FC pretrained model config name
W2V2FC_CONFIG = 'charsiu/en_w2v2_fc_10ms'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

#Window size of the model
WINDOW_SIZE = 400


###############################################################################
# Preprocess Charsiu latents
###############################################################################


def from_audios(
    audio,
    lengths,
    sample_rate=ppgs.SAMPLE_RATE,
    config=W2V2FC_CONFIG,
    gpu=None):
    """Compute W2V2FC latents from audio"""

    with torch.no_grad():
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audio, 'model'):
            from_audio.model = transformers.Wav2Vec2Model.from_pretrained(
                config).to(device)

        # Maybe resample
        audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).to(device)
        pad = WINDOW_SIZE // 2 - ppgs.HOPSIZE // 2
        padded_audio = torch.nn.functional.pad(
            audio,
            (pad, pad)
        ).squeeze(dim=1)

        # Infer W2V2FC latents
        mask = ppgs.model.transformer.mask_from_lengths(
            lengths
        ).squeeze(dim=1).to(torch.long).to(audio.device)
        output = from_audio.model(padded_audio, mask).last_hidden_state
        output = torch.transpose(output, 1, 2)
        return output.to(torch.float16)


def from_audio(
    audio,
    sample_rate=ppgs.SAMPLE_RATE,
    config=W2V2FC_CONFIG,
    gpu=None):
    """Compute W2V2FC latents from audio"""

    with torch.no_grad():
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audio, 'model'):
            from_audio.model = transformers.Wav2Vec2Model.from_pretrained(
                config).to(device)

        # Maybe resample
        audio = ppgs.resample(audio, sample_rate, SAMPLE_RATE).squeeze()

        # Setup features
        pad = WINDOW_SIZE // 2 - ppgs.HOPSIZE // 2
        inputs = torch.nn.functional.pad(audio, (pad, pad)).unsqueeze(dim=0)
        inputs = inputs.to(device)

        # Infer W2V2FC latents
        with torch.no_grad():
            return from_audio.model(inputs).last_hidden_state.squeeze().T


def from_file(audio_file, gpu=None):
    """Compute W2V2FC latents from audio file"""
    return from_audio(ppgs.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute W2V2FC latents from audio file and save to disk"""
    ppg = from_file(audio_file, gpu).to(torch.float16)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute W2V2FC latents from audio files and save to disk"""
    for audio_file, output_file in torchutil.iterator(
        zip(audio_files, output_files),
        'Extracting W2V2FC latents',
        total=len(audio_files)
    ):
        from_file_to_file(audio_file, output_file, gpu)
