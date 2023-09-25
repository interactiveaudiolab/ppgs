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

`pip install ppgs`


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
    representation
        The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
    checkpoint
        Path to the checkpoint to use
    gpu
        The gpu to use for preprocessing

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
        Path to audio file
    representation
        The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
    checkpoint
        Path to the checkpoint to use
    gpu
        The gpu to use for preprocessing

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
        Path to audio file
    output_file
        Path to output file (ideally '.pt')
    representation
        The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
    preprocess_only
        Shortcut to just doing preprocessing for the given representation
    checkpoint
        Path to the checkpoint to use
    gpu
        The gpu to use for preprocessing
"""
```


#### `ppgs.from_files_to_files`

```
"""Infer ppgs from audio files and save to torch tensor files

Arguments
    audio_files
        Path to audio files
    output
        A list of output files or a path to an output directory
        If not provided, ppgs will be stored in same locations as audio files
    representation
        The type of latents to use (e.g. Wav2Vec 2.0 Facebook = 'w2v2fb')
    checkpoint
        Path to the checkpoint to use
    save_intermediate_features
        Saves the intermediate features (e.g. Wav2Vec 2.0 latents) in addition to ppgs
    gpu
        The gpu to use for preprocessing
"""
```

#### Command-line interface (CLI)

```
Compute phonetic posteriorgram (PPG) features

python -m ppgs
    --sources <list of files and/or directories> \
    --sinks <corresponding list of output files and/or directories> \
    --num-workers <number of CPU workers> \
    --gpu <gpu number>
```

## Training

### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

```
python -m ppgs.data.download --datasets <datasets>
```


### Preprocess

Prepares features for training. Features are stored in `data/cache/`.
wav and phoneme (ground truth alignment) features must be preprocessed first


```
python -m ppgs.data.preprocess \
   --datasets <datasets> \
   --gpu <gpu> \
   --num-workers <workers> \
   --features <latent features>
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
python -m ppgs.train \
    --config <config> \
    --dataset <dataset> \
    --gpus <gpus>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Monitor

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/ --port <port>
```


### Evaluate

Performs objective evaluation.
Also performs benchmarking of speed. Results are stored in `eval/`.

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
