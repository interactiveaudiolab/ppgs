import functools
import torch
import torchaudio
import tqdm

import ppgs


###############################################################################
# API
###############################################################################

#TODO add from_features
def from_audio(
    audio,
    sample_rate,
    model=None,
    representation=ppgs.REPRESENTATION,
    preprocess_only=False,
    checkpoint=None,
    gpu=None):
    """Compute phonetic posteriorgram features from audio"""
    with torch.no_grad():
        if checkpoint is None:
            checkpoint = ppgs.CHECKPOINT_DIR / f'{representation}.pt'

        if model is None:
            model = ppgs.Model()()

        # Cache model on first call; update when GPU or checkpoint changes
        if (not hasattr(from_audio, 'model') or
            from_audio.checkpoint != checkpoint or
            from_audio.gpu != gpu):
            # model = ppgs.model.BaselineModel()
            state_dict = torch.load(checkpoint, map_location='cpu')['model']
            try:
                model.load_state_dict(state_dict=state_dict)
            except RuntimeError:
                print('Failed to load model, trying again with assumption that model was trained using ddp')
                state_dict = ppgs.load.ddp_to_single_state_dict(state_dict)
                model.load_state_dict(state_dict)
            device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
            from_audio.model = model.to(device)
            from_audio.checkpoint = checkpoint
            from_audio.gpu = gpu

        # Preprocess audio
        features = ppgs.preprocess.from_audio(audio, representation=representation, sample_rate=sample_rate, gpu=gpu)

        if preprocess_only:
            return features

        # Compute PPGs
        return from_audio.model(features[None])[0]


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
    datasets,
    from_features,
    save_from_features,
    cache_dir,
    num_workers,
    gpu=None):
    """Process datasets using ppgs models (and the corresponding feature models)"""
    ppgs.CACHE_DIR = cache_dir
    for dataset in datasets:
        ppgs.preprocess.accel.multiprocessed_process(
            dataset,
            ppgs.CACHE_DIR / dataset,
            from_features,
            save_from_features,
            num_workers,
            gpu
        )