python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/bottleneck/00200000.pt --config runs/bottleneck/bottleneck.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fs/00200000.pt --config runs/w2v2fs/w2v2fs.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/w2v2fb/00200000.pt --config runs/w2v2fb/w2v2fb.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/mel/00200000.pt --config runs/mel/mel.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/unfold/00200000.pt --config runs/unfold/unfold.py --gpu $1 --partition test
python -m ppgs.evaluate --datasets timit arctic --checkpoint runs/encodec/00200000.pt --config runs/encodec/encodec.py --gpu $1 --partition test