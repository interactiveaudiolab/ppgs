import json
import random

import ppgs


def datasets(datasets, overwrite=False):
    """Partition datasets"""
    for dataset in datasets:

        # Check if partition already exists
        file = ppgs.PARTITION_DIR / f'{dataset}.json'
        if file.exists():
            if not overwrite:
                print(f'Not overwriting existing partition {file}')
                continue

        # Random seed
        random.seed(ppgs.RANDOM_SEED)

        if dataset.lower() == 'arctic':
            arctic()
        elif dataset.lower() == 'timit':
            timit()
        else:
            raise NotImplementedError

#TODO check if SA1, SA2 should be filtered out
def partition_dataset(data_dir, unseen_speakers, partition_file):
    unseen_speakers = set(unseen_speakers)
    all_textgrid = list(data_dir.rglob('*.textgrid'))
    train = (f'{file.parents[0].name}/{file.stem}' for file in all_textgrid if file.parents[0].name not in unseen_speakers)
    test = (f'{file.parents[0].name}/{file.stem}' for file in all_textgrid if file.parents[0].name in unseen_speakers)
    partition = {'train': list(train), 'test': list(test)}
    partition_file.parents[0].mkdir(parents=True, exist_ok=True)
    with open(partition_file, 'w') as f:
        json.dump(partition, f, ensure_ascii=False, indent=4)


def arctic():
    data_dir = ppgs.CACHE_DIR / 'arctic'
    unseen_speakers = [f'cmu_us_{speaker}_arctic' for speaker in ppgs.ARCTIC_UNSEEN]
    partition_file = ppgs.PARTITION_DIR / 'arctic.json'
    partition_dataset(data_dir, unseen_speakers, partition_file)

def timit():
    data_dir = ppgs.CACHE_DIR / 'timit'
    unseen_speakers = ppgs.TIMIT_UNSEEN
    partition_file = ppgs.PARTITION_DIR / 'timit.json'
    partition_dataset(data_dir, unseen_speakers, partition_file)