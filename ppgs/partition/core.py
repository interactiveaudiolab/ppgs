import json
import random

import ppgs


def datasets(datasets, overwrite=False, for_testing=False):
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
            arctic(for_testing)
        elif dataset.lower() == 'timit':
            timit(for_testing)
        elif dataset.lower() == 'charsiu':
            charsiu()
        else:
            raise NotImplementedError

def partition_dataset(data_dir, unseen_speakers, validation_files, partition_file, for_testing=False):
    unseen_speakers = set(unseen_speakers)
    all_textgrid = list(data_dir.rglob('*.textgrid')) #TODO merge with wav rglob
    if for_testing:
        test = [f'{file.parents[0].name}/{file.stem}' for file in all_textgrid]
        train = []
        valid = []
    else:
        valid = [f'{file.parents[0].name}/{file.stem}' for file in all_textgrid if file in validation_files]
        train = [f'{file.parents[0].name}/{file.stem}' for file in all_textgrid if (file.parents[0].name not in unseen_speakers) and file not in validation_files]
        test = [f'{file.parents[0].name}/{file.stem}' for file in all_textgrid if (file.parents[0].name in unseen_speakers) and file not in validation_files]
    partition = {'train': train, 'valid': valid, 'test': test}
    partition_file.parents[0].mkdir(parents=True, exist_ok=True)
    with open(partition_file, 'w') as f:
        json.dump(partition, f, ensure_ascii=False, indent=4)


def arctic(for_testing=False):
    data_dir = ppgs.CACHE_DIR / 'arctic'
    unseen_speakers = [f'cmu_us_{speaker}_arctic' for speaker in ppgs.ARCTIC_UNSEEN]
    partition_file = ppgs.PARTITION_DIR / 'arctic.json'
    
    #Get validation files
    validation_files = []
    for id in ppgs.ARCTIC_VALIDATION_IDS:
        validation_files += [file for file in data_dir.rglob(f'{id}.textgrid') if file.parent.name[7:10] not in ppgs.ARCTIC_UNSEEN]

    partition_dataset(data_dir, unseen_speakers, validation_files, partition_file, for_testing)

def timit(for_testing=False):
    data_dir = ppgs.CACHE_DIR / 'timit'
    unseen_speakers = ppgs.TIMIT_UNSEEN
    partition_file = ppgs.PARTITION_DIR / 'timit.json'

    #Get validation files
    validation_files = []
    for speaker in ppgs.TIMIT_VALID_SPEAKERS:
        validation_files += list((data_dir / speaker).glob('*.textgrid'))

    partition_dataset(data_dir, unseen_speakers, validation_files, partition_file, for_testing)

def charsiu():
    """Partition dataset"""
    data_dir = ppgs.CACHE_DIR / 'charsiu'
    partition_file = ppgs.PARTITION_DIR / 'charsiu.json'
    stems = [file.stem for file in data_dir.glob('*.textgrid')]
    stems = [stem for stem in stems if stem not in ppgs.CHARSIU_REJECT]
    # Get dataset stems
    random.seed(ppgs.RANDOM_SEED)
    random.shuffle(stems)

    # Get split points
    left, right = int(.70 * len(stems)), int(.85 * len(stems))

    # Perform partition
    partition = {
        'train': sorted(stems[:left]),
        'valid': sorted(stems[left:right]),
        'test': sorted(stems[right:])}

    # Write partition file
    with open(partition_file, 'w') as file:
        json.dump(partition, file, indent=4)