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
#  most single SSDs
python -m ppgs.preprocess --datasets charsiu --features w2v2fs --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerw2v2fs.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fs --force

python -m ppgs.preprocess --datasets charsiu --features w2v2fb --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerw2v2fb.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features w2v2fb --force

python -m ppgs.preprocess --datasets charsiu --features bottleneck --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerbottleneck.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features bottleneck --force

python -m ppgs.preprocess --datasets charsiu --features unfold --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerunfold.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features unfold --force

python -m ppgs.preprocess --datasets charsiu --features mel --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformermel.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features mel --force

python -m ppgs.preprocess --datasets charsiu --features spectrogram --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerspectrogram.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features spectrogram --force

python -m ppgs.preprocess --datasets charsiu --features encodec --gpu $1 --use-cached-inputs --num-workers $2
python -m ppgs.train --dataset charisu --config ppgs/assets/configs/transformerencodec.py --gpus $1
python -m ppgs.data.purge --datasets charsiu --features encodec --force

# Downlaod evaluation datasets
python -m ppgs.data.download --datasets timit arctic
python -m ppgs.preprocess --datasets timit arctic --features wav phonemes
python -m ppgs.preprocess --datasets timit arctic --features bottleneck w2v2fs w2v2fb mel spectrogram unfold encodec
python -m ppgs.partition --datasets timit arctic --overwrite --for-testing

python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerbottleneck/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerw2v2fs/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerw2v2fb/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerspectrogram/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformermel/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerunfold/300000.pt --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/transformerencodec/300000.pt --gpu $1 --partition test