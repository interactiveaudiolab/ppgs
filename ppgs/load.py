import json

import ppgs


###############################################################################
# Loading utilities
###############################################################################

def partition(dataset):
    """Load partitions for dataset"""
    with open(ppgs.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
