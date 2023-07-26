<h1 align="center">Phonetic Posteriorgrams (PPGS)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promonet)](https://pepy.tech/project/promonet)

</div>

Training, evaluation, and inference of neural phonetic posteriorgrams (PPGs) in PyTorch. Includes the original code for the paper _Disentangling Speech with Phonetic Posteriorgrams_. [[Paper]](TODO) [[Website]](TODO)


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
"""TODO

Args:
    TODO

Returns:
    TODO
"""
```


#### `ppgs.from_file`

```
"""TODO

Args:
    TODO

Returns:
    TODO
"""
```


#### `ppgs.from_file_to_file`

```
"""TODO

Args:
    TODO

Returns:
    TODO
"""
```


#### `ppgs.from_files_to_files`

```
"""TODO

Args:
    TODO

Returns:
    TODO
"""
```

#### Command-line interface (CLI)

**TODO** - use `python -m ppgs -h` and edit to look like, e.g., `penn` example

```
Compute phonetic posteriorgram (PPG) features

python -m ppgs
    --sources <list of files or directories> \
    --output <output files or directory> \
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

TODO - update this to be a single command

```
python -m ppgs.data.preprocess --datasets <datasets> --features wav phonemes
python -m ppgs.data.preprocess \
   --datasets <datasets> \
   --gpu <gpu> \
   --num-workers <workers> \
   --use-cached-inputs \
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
C. Churchwell, M. Morrison, and B. Pardo, "Disentangling Speech with Phonetic Posteriorgrams," Submitted
to ICASSP 2024, January 2024.


### BibTex

```
@inproceedings{churchwell2024disentangling,
    title={Disentangling Speech with Phonetic Posteriorgrams},
    author={Churchwell, Cameron and Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICASSP 2024},
    month={January},
    year={2024}
}
```

## Citation

### IEEE
C. Churchwell, M. Morrison, and B. Pardo, "Disentangling Speech with Phonetic Posteriorgrams," Submitted
to ICASSP 2024, January 2024.


### BibTex

```
@inproceedings{churchwell2024disentangling,
    title={Disentangling Speech with Phonetic Posteriorgrams},
    author={Churchwell, Cameron and Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICASSP 2024},
    month={January},
    year={2024}
}
```