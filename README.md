<h1 align="center">High-Fidelity Neural Phonetic Posteriorgrams</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ppgs.svg)](https://pypi.python.org/pypi/ppgs)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/ppgs)](https://pepy.tech/project/ppgs)

Training, evaluation, and inference of neural phonetic posteriorgrams (PPGs) in PyTorch

[[Paper]](https://www.maxrmorrison.com/pdfs/churchwell2024high.pdf) [[Website]](https://www.maxrmorrison.com/sites/ppgs/)
</div>


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface (API)](#application-programming-interface-api)
        * [`ppgs.from_audio`](#ppgsfrom_audio)
        * [`ppgs.from_file`](#ppgsfrom_file)
        * [`ppgs.from_file_to_file`](#ppgsfrom_file_to_file)
        * [`ppgs.from_files_to_files`](#ppgsfrom_files_to_files)
    * [Command-line interface (CLI)](#command-line-interface-cli)
- [Distance](#distance)
- [Interpolate](#interpolate)
- [Edit](#edit)
    * [`ppgs.edit.grid.constant`](#ppgseditgridconstant)
    * [`ppgs.edit.grid.from_alignments`](#ppgseditgridfrom_alignments)
    * [`ppgs.edit.grid.of_length`](#ppgseditgridof_length)
    * [`ppgs.edit.grid.sample`](#ppgseditgridsample)
    * [`ppgs.edit.reallocate`](#ppgseditreallocate)
    * [`ppgs.edit.regex`](#ppgseditregex)
    * [`ppgs.edit.shift`](#ppgseditshift)
    * [`ppgs.edit.swap`](#ppgseditswap)
- [Sparsify](#sparsify)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
    * [Evaluate](#evaluate)
- [Citation](#citation)


## Installation

An inference-only installation with our best model is pip-installable

`pip install ppgs`

To perform training, install training dependencies and FFMPEG.

```bash
pip install ppgs[train]
conda install -c conda-forge ffmpeg
```

If you wish to use the Charsiu representation, download the code,
install both inference and training dependencies, and install
Charsiu as a Git submodule.

```bash
# Clone
git clone git@github.com/interactiveaudiolab/ppgs
cd ppgs/

# Install dependencies
pip install -e .[train]
conda install -c conda-forge ffmpeg

# Download Charsiu
git submodule init
git submodule update
```


## Inference

```python
import ppgs

# Load speech audio at correct sample rate
audio = ppgs.load.audio(audio_file)

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = 0

# Infer PPGs
ppgs = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu)
```


### Application programming interface (API)

#### `ppgs.from_audio`

```python
def from_audio(
    audio: torch.Tensor,
    sample_rate: Union[int, float],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: int = None
) -> torch.Tensor:
    """Infer ppgs from audio

    Arguments
        audio
            Batched audio to process
            shape=(batch, 1, samples)
        sample_rate
            Audio sampling rate
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            Phonetic posteriorgrams
            shape=(batch, len(ppgs.PHONEMES), frames)
    """
```


#### `ppgs.from_file`

```python
def from_file(
    file: Union[str, bytes, os.PathLike],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Infer ppgs from an audio file

    Arguments
        file
            The audio file
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference

    Returns
        ppgs
            Phonetic posteriorgram
            shape=(len(ppgs.PHONEMES), frames)
    """
```


#### `ppgs.from_file_to_file`

```python
def from_file_to_file(
    audio_file: Union[str, bytes, os.PathLike],
    output_file: Union[str, bytes, os.PathLike],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Infer ppg from an audio file and save to a torch tensor file

    Arguments
        audio_file
            The audio file
        output_file
            The .pt file to save PPGs
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        gpu
            The index of the GPU to use for inference
    """
```


#### `ppgs.from_files_to_files`

```python
def from_files_to_files(
    audio_files: List[Union[str, bytes, os.PathLike]],
    output_files: List[Union[str, bytes, os.PathLike]],
    representation: str = ppgs.REPRESENTATION,
    checkpoint: Optional[Union[str, bytes, os.PathLike]] = None,
    num_workers: int = 0,
    gpu: Optional[int] = None,
    max_frames: int = ppgs.MAX_INFERENCE_FRAMES
) -> None:
    """Infer ppgs from audio files and save to torch tensor files

    Arguments
        audio_files
            The audio files
        output_files
            The .pt files to save PPGs
        representation
            The representation to use, 'mel' and 'w2v2fb' are currently supported
        checkpoint
            The checkpoint file
        num_workers
            Number of CPU threads for multiprocessing
        gpu
            The index of the GPU to use for inference
        max_frames
            The maximum number of frames on the GPU at once
    """
```


### Command-line interface (CLI)

```
usage: python -m ppgs
    [-h]
    [--audio_files AUDIO_FILES [AUDIO_FILES ...]]
    [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]
    [--representation REPRESENTATION]
    [--checkpoint CHECKPOINT]
    [--num-workers NUM_WORKERS]
    [--gpu GPU]
    [--max-frames MAX_TRAINING_FRAMES]

arguments:
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
        Paths to input audio files
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
        The one-to-one corresponding output files

optional arguments:
    -h, --help
        Show this help message and exit
    --representation REPRESENTATION
        Representation to use for inference
    --checkpoint CHECKPOINT
        The checkpoint file
    --num-workers NUM_WORKERS
        Number of CPU threads for multiprocessing
    --gpu GPU
        The index of the GPU to use for inference. Defaults to CPU.
    --max-frames MAX_FRAMES
        Maximum number of frames in a batch
```


## Distance

To compute the proposed normalized Jenson-Shannon divergence pronunciation
distance between two PPGs, use `ppgs.distance()`.

```python
def distance(
    ppgX: torch.Tensor,
    ppgY: torch.Tensor,
    reduction: str = 'mean',
    normalize: bool = True,
    exponent: float = ppgs.SIMILARITY_EXPONENT
) -> torch.Tensor:
    """Compute the pronunciation distance between two aligned PPGs

    Arguments
        ppgX
            Input PPG X
            shape=(len(ppgs.PHONEMES), frames)
        ppgY
            Input PPG Y to compare with PPG X
            shape=(len(ppgs.PHONEMES), frames)
        reduction
            Reduction to apply to the output. One of ['mean', 'none', 'sum'].
        normalize
            Apply similarity based normalization
        exponent
            Similarty exponent

    Returns
        Normalized Jenson-shannon divergence between PPGs
    """
```


## Interpolate

```python
def interpolate(
    ppgX: torch.Tensor,
    ppgY: torch.Tensor,
    interp: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Linear interpolation

    Arguments
        ppgX
            Input PPG X
            shape=(len(ppgs.PHONEMES), frames)
        ppgY
            Input PPG Y
            shape=(len(ppgs.PHONEMES), frames)
        interp
            Interpolation values
            scalar float OR shape=(frames,)

    Returns
        Interpolated PPGs
        shape=(len(ppgs.PHONEMES), frames)
    """
```


## Edit

```python
import ppgs

# Get PPGs to edit
ppg = ppgs.from_file(audio_file, gpu=gpu)

# Constant-ratio time-stretching (slowing down)
grid = ppgs.edit.grid.constant(ppg, ratio=0.8)
slow = ppgs.edit.grid.sample(ppg, grid)

# Stretch to a desired length (e.g., 100 frames)
grid = ppgs.edit.grid.of_length(ppg, 100)
fixed = ppgs.edit.grid.sample(ppg, grid)
```


### `ppgs.edit.grid.constant`

```python
def constant(ppg: torch.Tensor, ratio: float) -> torch.Tensor:
    """Create a grid for constant-ratio time-stretching

    Arguments
        ppg
            Input PPG
        ratio
            Time-stretching ratio; lower is slower

    Returns
        Constant-ratio grid for time-stretching ppg
    """
```


### `ppgs.edit.grid.from_alignments`

```python
def from_alignments(
    source: pypar.Alignment,
    target: pypar.Alignment,
    sample_rate: int = ppgs.SAMPLE_RATE,
    hopsize: int = ppgs.HOPSIZE
) -> torch.Tensor:
    """Create time-stretch grid to convert source alignment to target

    Arguments
        source
            Forced alignment of PPG to stretch
        target
            Forced alignment of target PPG
        sample_rate
            Audio sampling rate
        hopsize
            Hopsize in samples

    Returns
        Grid for time-stretching source PPG
    """
```


### `ppgs.edit.grid.of_length`

```python
def of_length(ppg: torch.Tensor, length: int) -> torch.Tensor:
    """Create time-stretch grid to resample PPG to a specified length

    Arguments
        ppg
            Input PPG
        length
            Target length

    Returns
        Grid of specified length for time-stretching ppg
    """
```


### `ppgs.edit.grid.sample`

```python
def grid_sample(ppg: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Grid-based PPG interpolation

    Arguments
        ppg
            Input PPG
        grid
            Grid of desired length; each item is a float-valued index into ppg

    Returns
        Interpolated PPG
    """
```


### `ppgs.edit.reallocate`

```python
def reallocate(
    ppg: torch.Tensor,
    source: str,
    target: str,
    value: Optional[float] = None
) -> torch.Tensor:
    """Reallocate probability from source phoneme to target phoneme

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        source
            Source phoneme
        target
            Target phoneme
        value
            Max amount to reallocate. If None, reallocates all probability.

    Returns
        Edited PPG
    """
```


### `ppgs.edit.regex`

```python
def regex(
    ppg: torch.Tensor,
    source_phonemes: List[str],
    target_phonemes: List[str]
) -> torch.Tensor:
    """Regex match and replace (via swap) for phoneme sequences

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        source_phonemes
            Source phoneme sequence
        target_phonemes
            Target phoneme sequence

    Returns
        Edited PPG
    """
```


### `ppgs.edit.shift`

```python
def shift(ppg: torch.Tensor, phoneme: str, value: float):
    """Shift probability of a phoneme and reallocate proportionally

    Arguments
        ppg
            Input PPG
            shape=(len(ppgs.PHONEMES), frames)
        phoneme
            Input phoneme
        value
            Maximal shift amount

    Returns
        Edited PPG
    """
```


### `ppgs.edit.swap`

```python
def swap(ppg: torch.Tensor, phonemeA: str, phonemeB: str) -> torch.Tensor:
    """Swap the probabilities of two phonemes

    Arguments
        ppg
            Input PPG
            shape=(len(ppg.PHONEMES), frames)
        phonemeA
            Input phoneme A
        phonemeB
            Input phoneme B

    Returns
        Edited PPG
    """
```


## Sparsify

```python
def sparsify(
    ppg: torch.Tensor,
    method: str = 'percentile',
    threshold: torch.Tensor = torch.Tensor([0.85])
) -> torch.Tensor:
    """Make phonetic posteriorgrams sparse

    Arguments
        ppg
            Input PPG
            shape=(batch, len(ppgs.PHONEMES), frames)
        method
            Sparsification method. One of ['constant', 'percentile', 'topk'].
        threshold
            In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.

    Returns
        Sparse phonetic posteriorgram
        shape=(batch, len(ppgs.PHONEMES), frames)
    """
```


## Training

### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

**N.B.** Common voice and TIMIT cannot be automatically downloaded. You must
manually download the tarballs and place them in `data/sources/commonvoice`
or `data/sources/timit`, respectively, prior to running the following.

```bash
python -m ppgs.data.download --datasets <datasets>
```


### Preprocess

Prepares representations for training. Representations are stored
in `data/cache/`.

```
python -m ppgs.preprocess \
   --datasets <datasets> \
   --representatations <representations> \
   --gpu <gpu> \
   --num-workers <workers>
```


### Partition

Partitions a dataset. You should not need to run this, as the partitions
used in our work are provided for each dataset in
`ppgs/assets/partitions/`.

```
python -m ppgs.partition --datasets <datasets>
```


### Train

Trains a model. Checkpoints and logs are stored in `runs/`.

```
python -m ppgs.train --config <config> --dataset <dataset> --gpu <gpu>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Monitor

You can monitor training via `tensorboard`.

```
tensorboard --logdir runs/ --port <port> --load_fast true
```

To use the `torchutil` notification system to receive notifications for long
jobs (download, preprocess, train, and evaluate), set the
`PYTORCH_NOTIFICATION_URL` environment variable to a supported webhook as
explained in [the Apprise documentation](https://pypi.org/project/apprise/).


### Evaluate

Performs objective evaluation of phoneme accuracy. Results are stored
in `eval/`.

```
python -m ppgs.evaluate \
    --config <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```


## Citation

### IEEE
C. Churchwell, M. Morrison, and B. Pardo, "High-Fidelity Neural Phonetic Posteriorgrams,"
ICASSP 2024 Workshop on Explainable Machine Learning for Speech and Audio, April 2024.


### BibTex

```
@inproceedings{churchwell2024high,
    title={High-Fidelity Neural Phonetic Posteriorgrams},
    author={Churchwell, Cameron and Morrison, Max and Pardo, Bryan},
    booktitle={ICASSP 2024 Workshop on Explainable Machine Learning for Speech and Audio},
    month={April},
    year={2024}
}
```
