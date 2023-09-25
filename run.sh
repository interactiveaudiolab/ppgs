# Runs all experiments in the paper
# "High-Fidelity Neural Phonetic Posteriorgrams"

# Args
# $1 - index of GPU to use
# $2 - number of threads for preprocessing


###############################################################################
# Setup training data
###############################################################################


python -m ppgs.download --datasets charsiu
python -m ppgs.preprocess --datasets charsiu --features wav phonemes
python -m ppgs.partition --datasets charsiu
python -m ppgs.data.purge --datasets charsiu --kinds sources datasets --force


###############################################################################
# Train
###############################################################################


# We purge cache data after each experiment because the dataset is large and
# the cache easily fills most SSDs
python -m ppgs.preprocess --datasets charsiu --features w2v2fc --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config config/w2v2fc.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fc --force

python -m ppgs.preprocess --datasets charsiu --features w2v2fb --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config config/w2v2fb.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fb --force

python -m ppgs.preprocess --datasets charsiu --features bottleneck --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config config/bottleneck.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features bottleneck --force

python -m ppgs.preprocess --datasets charsiu --features mel --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config config/mel.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features mel --force

python -m ppgs.preprocess --datasets charsiu --features encodec --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config config/encodec.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features encodec --force


###############################################################################
# Setup evaluation data (TODO - interleave with training)
###############################################################################


python -m ppgs.data.download --datasets timit arctic
python -m ppgs.preprocess --datasets timit arctic --features wav phonemes
python -m ppgs.partition --datasets timit arctic --overwrite --for-testing


###############################################################################
# Evaluate (TODO - interleave with training)
###############################################################################


python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/bottleneck/00200000.pt --config config/bottleneck.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fc/00200000.pt --config config/w2v2fc.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fb/00200000.pt --config config/w2v2fb.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/mel/00200000.pt --config config/mel.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/encodec/00200000.pt --config config/encodec.py --gpu $1 --partition test
