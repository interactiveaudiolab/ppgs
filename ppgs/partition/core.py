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
            partition = arctic()
        else:
            raise NotImplementedError

        # Save to disk
        # file.parent.mkdir(exist_ok=True, parents=True)
        # with open(file, 'w') as file:
        #     json.dump(partition, file, ensure_ascii=False, indent=4)

def arctic():
    # Get list of speakers
    directory = ppgs.CACHE_DIR / 'arctic'
    # stems = {
    #     f'{file.parent.name}/{file.stem[:-4]}'
    #     for file in directory.rglob('*.json')}
    # stems = {
    #     f'{file.parent.name}/{file.stem}'
    #     for file in directory.rglob('*')
    # }

    speakers = [file.stem for file in directory.glob('*')]

    for speaker in speakers:
        if speaker[7:10] in ppgs.ARCTIC_UNSEEN: #test split
            pass
        else: #train split
            pass

def TIMIT():
    pass