import functools
import torch
import torchaudio
import tqdm
from pathlib import Path
import ppgs
from typing import List
from ppgs.data import aggregate

###############################################################################
# API
###############################################################################

def from_features(
    features: torch.Tensor,
    new_lengths: torch.Tensor,
    representation=ppgs.REPRESENTATION,
    checkpoint=None,
    gpu=None
):
    """Compute PPGs from features given by the representation"""
    with torch.inference_mode():
        return ppgs.REPRESENTATION_MAP[representation].from_features(
            features, 
            new_lengths,
            checkpoint, 
            gpu,
        )

def from_audio(
    audio,
    sample_rate,
    representation=ppgs.REPRESENTATION,
    preprocess_only=False,
    checkpoint=None,
    gpu=None):
    """Compute phonetic posteriorgram features from audio"""
    with torch.inference_mode():
        if checkpoint is None:
            checkpoint = ppgs.CHECKPOINT_DIR / f'{representation}.pt'

        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
        # Cache model on first call; update when GPU or checkpoint changes
        if (not hasattr(from_audio, 'model') or
            from_audio.checkpoint != checkpoint or
            from_audio.gpu != gpu):

            from_audio.model = ppgs.Model()()

            state_dict = torch.load(checkpoint, map_location='cpu')['model']

            try:
                from_audio.model.load_state_dict(state_dict=state_dict)

            except RuntimeError:
                print('Failed to load model, trying again with assumption that model was trained using ddp')
                state_dict = ppgs.load.ddp_to_single_state_dict(state_dict)
                from_audio.model.load_state_dict(state_dict)

            from_audio.model = from_audio.model.to(device)

            from_audio.checkpoint = checkpoint
            from_audio.gpu = gpu

        #TODO just use from_features

        # Preprocess audio
        features = ppgs.preprocess.from_audio(audio, representation=representation, sample_rate=sample_rate, gpu=gpu)

        if preprocess_only:
            return features

        if features.dim() == 2:
            features = features[None]

        # Compute PPGs
        if ppgs.MODEL == 'convolution':
            return from_audio.model(features)[0]
        else:
            return from_audio.model(features, torch.tensor([features.shape[-1]], device=device))[0]


def from_file(
        file,
        representation=ppgs.REPRESENTATION,
        preprocess_only=False, 
        checkpoint=None, 
        gpu=None
    ):
    """Compute phonetic posteriorgram features from audio file"""
    # Load audio
    audio = ppgs.load.audio(file)

    # Compute PPGs
    return from_audio(audio, sample_rate=ppgs.SAMPLE_RATE, representation=representation, preprocess_only=preprocess_only, checkpoint=checkpoint, gpu=gpu)


def from_file_to_file(
    audio_file,
    output_file,
    representation=ppgs.REPRESENTATION,
    preprocess_only=False,
    checkpoint=None,
    gpu=None):
    """Compute phonetic posteriorgram and save as torch tensor"""
    # Compute PPGs
    result = from_file(audio_file, representation=representation, preprocess_only=preprocess_only, checkpoint=checkpoint, gpu=gpu).detach().cpu()

    # Save to disk
    torch.save(result, output_file)


def from_files_to_files(
    audio_files,
    output_files=None,
    representation=None,
    preprocess_only=False,
    checkpoint=None,
    gpu=None):
    """Compute phonetic posteriorgrams and save as torch tensors"""
    # Default output files are audio paths with ".pt" extension
    if output_files is None:
        output_files = [file.with_suffix('.pt') for file in audio_files]

    # Bind common parameters
    ppg_fn = functools.partial(
        from_file_to_file,
        representation=representation,
        preprocess_only=preprocess_only,
        checkpoint=checkpoint,
        gpu=gpu)

    # Compute PPGs
    iterable = iterator(
        zip(audio_files, output_files),
        'ppgs',
        total=len(audio_files))
    for audio_file, output_file in iterable:
        ppg_fn(audio_file, output_file)


###############################################################################
# Utilities
###############################################################################


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


def resample(audio, sample_rate, target_rate=ppgs.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)


def process(
    sources: List[Path],
    from_feature=ppgs.REPRESENTATION,
    save_intermediate_features=False,
    output: Path = None,
    num_workers=2,
    gpu=None):
    """Process datasets using ppgs models (and the corresponding feature models)"""
    files = aggregate(sources, ['.wav', '.mp3'])
    ppgs.preprocess.accel.multiprocessed_process(
        files,
        [from_feature],
        save_intermediate_features,
        output,
        num_workers,
        gpu
    )