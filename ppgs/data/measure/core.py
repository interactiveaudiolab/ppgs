from math import ceil

import tqdm

import ppgs


def datasets(datasets, features=None, unit='B'):
    """Purge datasets from local memory"""
    if features is None:
        features = ppgs.preprocess.ALL_FEATURES
    total = 0
    for dataset in datasets:

        subtotal = 0
        for feature in features:
            if feature in ppgs.REPRESENTATION_MAP.keys():
                subtotal += measure_glob(ppgs.CACHE_DIR / dataset, f'**/*-{feature}.pt', unit)
            elif feature == 'wav':
                subtotal += measure_glob(ppgs.CACHE_DIR / dataset, '**/*.wav', unit)
            elif feature == 'phonemes':
                subtotal += measure_glob(ppgs.CACHE_DIR / dataset, '**/*.textgrid', unit)
        print(f"total for dataset {dataset} is {file_size_to_string(subtotal, unit)}")
        total += subtotal
    print(f"total is {file_size_to_string(total, unit)}")


def measure_glob(path, glob_string, unit='B'):
    print(f"collecting files in glob {path}/{glob_string}")
    files = list(path.glob(glob_string))
    total = 0
    iterator = tqdm.tqdm(
        files,
        "measuring files for glob: " + glob_string,
        len(files),
        dynamic_ncols=True
    )
    for file in iterator:
        total += file.stat().st_size
    print(f"total for glob \"{glob_string}\" is {file_size_to_string(total, unit)}")
    return total

exponent_map = {
    'B': 0,
    'KB': 1,
    'MB': 2,
    'GB': 3,
    'TB': 4,
}
def file_size_to_string(size_in_bytes, new_unit='B'):
    exponent = exponent_map[new_unit]
    divisor = 1024 ** exponent
    return f"{ceil(size_in_bytes / divisor)} {new_unit}"