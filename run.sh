# Runs all experiments in the paper
# "High-Fidelity Neural Phonetic Posteriorgrams"

# Usage
# CUDA_VISIBLE_DEVICES=<gpus> ./run.sh


###############################################################################
# Setup data
###############################################################################


python -m ppgs.data.download
python -m ppgs.partition


###############################################################################
# Run experiments
###############################################################################


# N.B. We purge cache data after each experiment because the dataset is large and
#      the cache easily fills most SSDs

# Charsiu
python -m ppgs.preprocess --representations w2v2fc --gpu 0
accelerate launch -m ppgs.train --config config/w2v2fc.py
python -m ppgs.evaluate \
    --config config/w2v2fc.py \
    --checkpoint runs/w2v2fc/00200000.pt \
    --gpu 0
python -m ppgs.data.purge --representations w2v2fc --force

# Wav2vec 2.0
python -m ppgs.preprocess --representations w2v2fb --gpu 0
accelerate launch -m ppgs.train --config config/w2v2fb.py
python -m ppgs.evaluate \
    --config config/w2v2fb.py \
    --checkpoint runs/w2v2fb/00200000.pt \
    --gpu 0
python -m ppgs.data.purge --representations w2v2fb --force

# ASR bottleneck
python -m ppgs.preprocess --representations bottleneck --gpu 0
accelerate launch -m ppgs.train --config config/bottleneck.py
python -m ppgs.evaluate \
    --config config/bottleneck.py \
    --checkpoint runs/bottleneck/00200000.pt \
    --gpu 0
python -m ppgs.data.purge --representations bottleneck --force

# Mel spectrogram
python -m ppgs.preprocess --representations mel --gpu 0
accelerate launch -m ppgs.train --config config/mel.py
python -m ppgs.evaluate \
    --config config/mel.py \
    --checkpoint runs/mel/00200000.pt \
    --gpu 0
python -m ppgs.data.purge --representations mel --force

# EnCodec
python -m ppgs.preprocess --representations encodec --gpu 0
accelerate launch -m ppgs.train --config config/encodec.py
python -m ppgs.evaluate \
    --config config/encodec.py \
    --checkpoint runs/encodec/00200000.pt \
    --gpu 0
python -m ppgs.data.purge --representations encodec --force


###############################################################################
# TODO - create accuracy plot
###############################################################################

