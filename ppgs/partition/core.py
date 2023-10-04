import json
import random

import ppgs


###############################################################################
# Constants
###############################################################################


# Charsiu samples to reject due to data errors
CHARSIU_REJECT = ['common_voice_en_26168033']


###############################################################################
# Partition datasets
###############################################################################


def datasets(datasets):
    """Partition datasets"""
    for dataset in datasets:

        # Deterministic partitioning
        random.seed(ppgs.RANDOM_SEED)

        if dataset == 'arctic':
            partition = arctic()
        elif dataset == 'timit':
            partition = timit()
        elif dataset == 'charsiu':
            partition = charsiu()
        else:
            raise NotImplementedError

        # Write partition file
        with open(ppgs.PARTITION_DIR / f'{dataset}.json', 'w') as file:
            json.dump(partition, file, indent=4)


###############################################################################
# Individual datasets
###############################################################################


def arctic():
    """Partition Arctic"""
    # Get stems
    stems = [
        f'{file.parents[0].name}/{file.stem}'
        for file in (ppgs.CACHE_DIR / 'arctic').rglob('*.TextGrid')]

    # Arctic is only used for evaluation
    return {'train': [], 'valid': [], 'test': stems}


def timit():
    """Partition TIMIT"""
    # Get stems
    stems = [
        f'{file.parents[0].name}/{file.stem}'
        for file in (ppgs.CACHE_DIR / 'timit').rglob('*.TextGrid')]

    # TIMIT is only used for evaluation
    return {'train': [], 'valid': [], 'test': stems}


def charsiu():
    """Partition dataset"""
    # Get stems
    stems = [
        file.stem for file in (ppgs.CACHE_DIR / 'charsiu').rglob('*.TextGrid')
        if file.stem not in CHARSIU_REJECT]
    random.shuffle(stems)

    # Get split points
    left, right = int(.80 * len(stems)), int(.90 * len(stems))

    # Partition
    return {
        'train': sorted(stems[:left]),
        'valid': sorted(stems[left:right]),
        'test': sorted(stems[right:])}
