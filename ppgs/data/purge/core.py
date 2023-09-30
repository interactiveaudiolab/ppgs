import os
import shutil
from pathlib import Path

import ppgs


###############################################################################
# Data cleanup
###############################################################################


def datasets(
    datasets,
    features=None,
    kinds=None,
    force=False):
    """Cleanup data"""
    # Handle defaults
    if features is None or features in ['all', '*']:
        features = ppgs.preprocess.ALL_FEATURES
    if kinds is None:
        kinds = ['cache']

    # Handle wildcard
    if kinds in ['all', '*']:
        kinds = ['cache', 'datasets', 'sources', 'partitions']

    # List of features
    all_features = (
        list(ppgs.ALL_REPRESENTATIONS) +
        [rep + '-ppg' for rep in ppgs.ALL_REPRESENTATIONS])

    # Initialize cleaner
    purger = Purger(force=force)

    for dataset in datasets:

        # Add cache files
        if 'cache' in kinds:
            for feature in features:

                # Input representations
                if feature in all_features:
                    purger.add_glob(
                        ppgs.CACHE_DIR / dataset,
                        f'**/*-{feature}.pt')

                # Audio
                elif feature == 'wav':
                    purger.add_glob(ppgs.CACHE_DIR / dataset, '**/*.wav')

                # Alignments
                elif feature == 'phonemes':
                    purger.add_glob(ppgs.CACHE_DIR / dataset, '**/*.TextGrid')

        # Add datasets
        if 'datasets' in kinds:
            purger.add_path(ppgs.DATA_DIR / dataset)

        # Add tarballs, zipfiles, etc.
        if 'sources' in kinds:
            purger.add_path(ppgs.SOURCES_DIR / dataset)

        # Add partitions
        if 'partitions' in kinds:
            purger.add_path(ppgs.PARTITION_DIR / (dataset + '.json'))

    # Cleanup
    purger.purge()


###############################################################################
# Utilities
###############################################################################


class Purger():
    """Helper class to create purge queues"""

    def __init__(self, force=False):
        self.force = force
        self.paths = []
        self.globs = []

    def add_path(self, path):
        self.paths.append(path)

    def add_paths(self, paths: list):
        self.paths = self.paths + paths

    def add_glob(self, path, glob):
        self.globs.append([path, glob])

    def purge(self):
        # Maybe prompt user for confirmation
        if not self.force:
            print('\n'.join([str(p) for p in self.paths]))
            print('\n'.join([str(Path(p) / g) for p, g in self.globs]))
            if input(
                'Are you sure that you want to COMPLETELY '
                'delete the above items? [y/N]:'
            ).lower() != 'y':
                raise RuntimeError('Purge aborted by user')

        # Add globs as paths
        for path, glob in self.globs:
            self.add_paths(list(Path(path).glob(glob)))

        # Purge
        for path in self.paths:
            path = Path(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
