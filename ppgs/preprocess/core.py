"""core.py - data preprocessing"""


import ppgs


###############################################################################
# Preprocess
###############################################################################


def datasets(datasets):
    """Preprocess a dataset

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    for dataset in datasets:
        input_directory = ppgs.DATA_DIR / dataset
        output_directory = ppgs.CACHE_DIR / dataset

        # TODO - Perform preprocessing
        raise NotImplementedError
