# Runs all experiments in the paper
# "TODO"

# Args
# $1 - index of GPU to use
# $2 number of threads for preprocessing

# Download datasets
python -m ppgs.download --datasets charsiu

# Setup experiments
python -m ppgs.preprocess --datasets charsiu --features wav phonemes
python -m ppgs.partition --datasets charsiu
python -m ppgs.data.purge --datasets charsiu --kinds sources datasets --force

# run experiments interleaved with preprocessing and purging.
#  we need to purge because the dataset is quite large and will easily fill
#  most SSDs
python -m ppgs.preprocess --datasets charsiu --features w2v2fs --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/w2v2fs.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fs --force

python -m ppgs.preprocess --datasets charsiu --features w2v2fb --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/w2v2fb.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fb --force

python -m ppgs.preprocess --datasets charsiu --features bottleneck --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/bottleneck.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features bottleneck --force

python -m ppgs.preprocess --datasets charsiu --features unfold --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/unfold.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features unfold --force

python -m ppgs.preprocess --datasets charsiu --features mel --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/mel.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features mel --force

python -m ppgs.preprocess --datasets charsiu --features encodec --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --config ppgs/assets/configs/encodec.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features encodec --force

# Downlaod evaluation datasets
python -m ppgs.data.download --datasets timit arctic
python -m ppgs.preprocess --datasets timit arctic --features wav phonemes
python -m ppgs.partition --datasets timit arctic --overwrite --for-testing

python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/bottleneck/00200000.pt --config runs/bottleneck/bottleneck.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fs/00200000.pt --config runs/w2v2fs/w2v2fs.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fb/00200000.pt --config runs/w2v2fb/w2v2fb.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/mel/00200000.pt --config runs/mel/mel.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/unfold/00200000.pt --config runs/unfold/unfold.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/encodec/00200000.pt --config runs/encodec/encodec.py --gpu $1 --partition test