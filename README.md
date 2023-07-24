<h1 align="center">Phonetic Posteriorgrams (PPGS)</h1>
<div align="center">

<!-- [![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promonet)](https://pepy.tech/project/promonet) -->

</div>

<!-- Official code for the paper _Adaptive Neural Speech Prosody Editing_
[[paper]](https://www.maxrmorrison.com/pdfs/morrison2023adaptive.pdf)
[[companion website]](https://www.maxrmorrison.com/sites/promonet/) -->


## Table of contents

- [Installation](#installation)
- [Usage](#usage)
    * [Example](#example)
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


## Usage

### Examples

To use `ppgs` you simply pass in audio files and a destination to save PPG tensor files (.pt) to.


#### Application programming interface (API)

```
import ppgs


# from audio files
audio_files = [...] # you define this
predicted_ppgs = ppgs.from_files_to_files(audio_files, gpu=0)

# from audio files to memory
predicted_ppgs = ppgs.from_files(audio_files)

# from audio tensors (e.g. in a pipeline/dataset)
audios = TODO (list OR tensor)
predicted_ppgs = ppgs.from_audio(audios)
```

#### Command-line interface (CLI)

```
Compute phonetic posteriorgram (PPG) features

python -m ppgs
    --sources <list of files or directory>
    --output <output files or directory>
    --num-workers <number of workers>
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
python -m ppgs.data.preprocess --datasets <datasets> --features wav phonemes
python -m ppgs.data.preprocess --datasets <datasets> --gpu <gpu> --num-workers <workers> --use-cached-inputs --features <latent features>
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
    --checkpoint <checkpoint>
    --gpus <gpus>
```

TODO add citation section