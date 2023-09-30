import math

import ppgs


###############################################################################
# Measure dataset sizes
###############################################################################


def datasets(datasets, features=None):
    """Get dataset sizes in gigabytes"""
    if features is None:
        features = ppgs.preprocess.ALL_FEATURES

    total = 0
    for dataset in datasets:
        print(f'Measuring {dataset}')

        # Measure one dataset
        subtotal = 0
        cache_directory = ppgs.CACHE_DIR / dataset
        for feature in features:

            # Measure one feature in one dataset
            if feature in ppgs.ALL_REPRESENTATIONS:
                feature_size = measure_glob(
                    cache_directory,
                    f'**/*-{feature}.pt')
            elif feature == 'wav':
                feature_size = measure_glob(cache_directory, '**/*.wav')
            elif feature == 'phonemes':
                feature_size = measure_glob(cache_directory, '**/*.TextGrid')
            print(f'\t{feature}: {size_to_string(feature_size)}')

            # Update dataset total
            subtotal += feature_size

        # Update aggregate total
        print(f'{dataset} is {size_to_string(subtotal)}')
        total += subtotal
    print(f'Total is {size_to_string(total)}')


###############################################################################
# Utilities
###############################################################################


def measure_glob(path, glob_string):
    """Get the size in bytes of all files matching glob"""
    return sum(file.stat().st_size for file in path.glob(glob_string))


def size_to_string(size_in_bytes):
    """Format size in gigabytes"""
    return f'{math.ceil(size_in_bytes / (1024 ** 3))} GB'
