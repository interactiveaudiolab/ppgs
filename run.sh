# Runs all experiments in the paper
# "High-Fidelity Neural Phonetic Posteriorgrams"

# Args
# $1 - index of GPU to use
# $2 - number of threads for preprocessing


###############################################################################
# Setup training data
###############################################################################


python -m ppgs.download
python -m ppgs.preprocess --features wav phonemes
python -m ppgs.partition
python -m ppgs.data.purge \
    --datasets charsiu \
    --kinds sources datasets \
    --force


###############################################################################
# Run experiments
###############################################################################


# We purge cache data after each experiment because the dataset is large and
# the cache easily fills most SSDs
python -m ppgs.preprocess --features w2v2fc --gpu $1 --num-workers $2
python -m ppgs.train --config config/w2v2fc.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fc --force

python -m ppgs.preprocess --datasets charsiu --features w2v2fb --gpu $1 --num-workers $2
python -m ppgs.train --config config/w2v2fb.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fb --force

python -m ppgs.preprocess --datasets charsiu --features bottleneck --gpu $1 --num-workers $2
python -m ppgs.train --config config/bottleneck.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features bottleneck --force

python -m ppgs.preprocess --datasets charsiu --features mel --gpu $1 --num-workers $2
python -m ppgs.train --config config/mel.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features mel --force

python -m ppgs.preprocess --datasets charsiu --features encodec --gpu $1 --num-workers $2
python -m ppgs.train --config config/encodec.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features encodec --force
