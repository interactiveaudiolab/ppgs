# Phonetic posteriorgrams

## Installation

`pip install -e .`

## Usage

### Download data

`python -m ppgs.data.download --datasets <datasets>`

All datasets are saved in `data/datasets/DATASET`, where `DATASET` is the name
of the dataset.


### Preprocess data

`python -m ppgs.preprocess --datasets <datasets> --gpu <gpu>`

All preprocessed data are saved in `data/cache/DATASET`.


### Partition data

`python -m ppgs.partition --datasets <datasets>`


### Train

Complete all TODOs in `data/` and `model.py`, then run `python -m ppgs.train --config <config> --dataset
DATASET --gpus <gpus>`.


### Evaluate

Complete all TODOs in `evaluate/`, then run `python -m ppgs.evaluate
--datasets <datasets> --checkpoint <checkpoint> --gpu <gpu>`.


### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


### Test

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
Complete all TODOs in `test_model.py` and `test_data.py`, then run `pytest`.
Adding project-specific tests for preprocessing, inference, and inference is
encouraged.
