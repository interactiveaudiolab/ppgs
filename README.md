<h1 align="center">Phonetic Posteriorgrams (PPGs)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/promonet)](https://pepy.tech/project/promonet)

</div>

Training, evaluation, and inference of neural phonetic posteriorgrams (PPGs) in PyTorch. Includes the original code for the paper _High-Fidelity Neural Phonetic Posteriorgrams_. [[Paper]](https://www.maxrmorrison.com/pdfs/churchwell2024high.pdf) [[Website]](https://www.maxrmorrison.com/sites/ppgs/)


## Table of contents

- [Installation](#installation)
- [Inference](#inference)
    * [Application programming interface (API)](#application-programming-interface-api)
        * [`ppgs.from_audio`](#ppgsfrom_audio)
        * [`ppgs.from_file`](#ppgsfrom_file)
        * [`ppgs.from_file_to_file`](#ppgsfrom_file_to_file)
        * [`ppgs.from_files_to_files`](#ppgsfrom_files_to_files)
    * [Command-line interface (CLI)](#command-line-interface-cli)
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

```
pip install ppgs[train]
conda install -c conda-forge 'ffmpeg<5'
``````

If you wish to use the Charsiu representation, download the code,
install both inference and training dependencies, and install
Charsiu as a Git submodule.

```bash
# Clone
git clone git@github.com/interactiveaudiolab/ppgs
cd ppgs/

# Install dependencies
pip install -e .[train]
conda install -c conda-forge 'ffmpeg<5'

# Download Charsiu
git submodule init
git submodule update
```


## Inference

```
import ppgs

# Load speech audio at correct sample rate
audio = ppgs.load.audio(audio_file)

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = 0

# Infer PPGs
ppgs = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu)
```

#### Application programming interface (API)

#### `ppgs.from_audio`

```
"""Infer ppgs from audio

Arguments
    audio
        The batched audio to process in the shape BATCH x 1 x TIME
    lengths
        The lengths of the features
    checkpoint
        The checkpoint file
    gpu
        The index of the GPU to use for inference

Returns
    ppgs
        A tensor encoding ppgs with shape BATCH x DIMS x TIME
"""
```


#### `ppgs.from_file`

```
"""Infer ppgs from an audio file

Arguments
    file
        The audio file
    representation
        The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
    checkpoint
        The checkpoint file
    gpu
        The index of the GPU to use for inference

Returns
    ppgs
        A tensor encoding ppgs with shape 1 x DIMS x TIME
"""
```


#### `ppgs.from_file_to_file`

```
"""Infer ppg from an audio file and save to a torch tensor file

Arguments
    audio_file
        The audio file
    output_file
        The .pt file to save PPGs
    checkpoint
        The checkpoint file
    gpu
        The index of the GPU to use for inference
"""
```


#### `ppgs.from_files_to_files`

```
"""Infer ppgs from audio files and save to torch tensor files

Arguments
    audio_files
        The audio files
    output_files
        The .pt files to save PPGs
    checkpoint
        The checkpoint file
    num_workers
        Number of CPU threads for multiprocessing
    gpu
        The index of the GPU to use for inference
"""
```


#### Command-line interface (CLI)

```
usage: python -m ppgs
    [-h]
    [--input_paths INPUT_PATHS [INPUT_PATHS ...]]
    [--output_paths OUTPUT_PATHS [OUTPUT_PATHS ...]]
    [--extensions EXTENSIONS [EXTENSIONS ...]]
    [--checkpoint CHECKPOINT]
    [--num-workers NUM_WORKERS]
    [--gpu GPU]

arguments:
    --input_paths INPUT_PATHS [INPUT_PATHS ...]
        Paths to audio files and/or directories
    --output_paths OUTPUT_PATHS [OUTPUT_PATHS ...]
        The one-to-one corresponding output paths

optional arguments:
    -h, --help
        Show this help message and exit
    --extensions EXTENSIONS [EXTENSIONS ...]
        Extensions to glob for in directories
    --checkpoint CHECKPOINT
        The checkpoint file
    --num-workers NUM_WORKERS
        Number of CPU threads for multiprocessing
    --gpu GPU
        The index of the GPU to use for inference. Defaults to CPU.
```


## Training

### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

**N.B.** Charsiu and TIMIT cannot be automatically downloaded. You must
manually download the tarballs and place them in `data/sources/charsiu`
or `data/sources/timit`, respectively, prior to running the following.

```
python -m ppgs.data.download --datasets <datasets>
```


### Preprocess

Prepares representations for training. Representations are stored
in `data/cache/`.

```
python -m ppgs.data.preprocess \
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

Trains a model. Checkpoints and logs are stored in `runs/`. You may want to run
`accelerate config` first to configure which devices are used for training.

```
CUDA_VISIBLE_DEVICES=<gpus> accelerate launch -m ppgs.train \
    --config <config> \
    --dataset <dataset>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.

You can monitor training via `tensorboard`.

```
tensorboard --logdir runs/ --port <port>
```


### Evaluate

Performs objective evaluation of phoneme accuracy. Results are stored
in `eval/`.

```
python -m ppgs.evaluate \
    --config <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --gpus <gpus>
```


## Citation

### IEEE
C. Churchwell, M. Morrison, and B. Pardo, "High-Fidelity Neural Phonetic Posteriorgrams," Submitted
to ICASSP 2024, April 2024.


### BibTex

```
@inproceedings{churchwell2024high,
    title={High-Fidelity Neural Phonetic Posteriorgrams},
    author={Churchwell, Cameron and Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICASSP 2024},
    month={April},
    year={2024}
}
```
