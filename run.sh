# Trains and evaluates all PPG models in the paper
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


# Run best model
python -m ppgs.preprocess --gpu 0
python -m ppgs.train --gpu 0
python -m ppgs.evaluate --gpu 0


###############################################################################
# Run experiments
###############################################################################


# Wav2vec 2.0
python -m ppgs.preprocess --representations w2v2fb --gpu 0
python -m ppgs.train --config config/w2v2fb.py --gpu 0
python -m ppgs.evaluate --config config/w2v2fb.py --gpu 0

# Charsiu
python -m ppgs.preprocess --representations w2v2fc --gpu 0
python -m ppgs.train --config config/w2v2fc.py --gpu 0
python -m ppgs.evaluate --config config/w2v2fc.py --gpu 0

# Mel spectrogram
python -m ppgs.preprocess --representations mel --gpu 0
python -m ppgs.train --config config/mel.py --gpu 0
python -m ppgs.evaluate --config config/mel.py --gpu 0

# ASR bottleneck
python -m ppgs.preprocess --representations bottleneck --gpu 0
python -m ppgs.train --config config/bottleneck.py --gpu 0
python -m ppgs.evaluate --config config/bottleneck.py --gpu 0

# EnCodec
python -m ppgs.preprocess --representations encodec --gpu 0
python -m ppgs.train --config config/encodec.py --gpu 0
python -m ppgs.evaluate --config config/encodec.py --gpu 0


###############################################################################
# Create accuracy plot
###############################################################################


python -m ppgs.plot.accuracy --output_file accuracy.pdf
