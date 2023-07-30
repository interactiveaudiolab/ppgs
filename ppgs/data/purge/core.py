import os
import shutil
from pathlib import Path

import ppgs


def datasets(datasets, features=None, kinds=None, force=False):
    """Purge datasets from local memory"""
    if features is None:
        features = ppgs.preprocess.ALL_FEATURES
    if kinds is None:
        kinds = ['cache']

    if 'all' in kinds:
        kinds = ['cache', 'datasets', 'sources', 'partitions']

    purger = Purger(force=force)

    for dataset in datasets:

        if 'cache' in kinds:
            for feature in features:
                if feature in list(ppgs.REPRESENTATION_MAP.keys()) + [rep + '-ppg' for rep in ppgs.REPRESENTATION_MAP.keys()]:
                    purger.add_glob(ppgs.CACHE_DIR / dataset, f'**/*-{feature}.pt')
                elif feature == 'wav':
                    purger.add_glob(ppgs.CACHE_DIR / dataset, '**/*.wav')
                elif feature == 'phonemes':
                    purger.add_glob(ppgs.CACHE_DIR / dataset, '**/*.textgrid')

        if 'datasets' in kinds:
            purger.add_path(ppgs.DATA_DIR / dataset)

        if 'sources' in kinds:
            purger.add_path(ppgs.SOURCES_DIR / dataset)

        if 'partitions' in kinds:
            purger.add_path(ppgs.PARTITION_DIR / (dataset + '.json'))

    purger.purge() #Prompt user (if force==True) and purge items in queue



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
        if not self.force:
            print('\n'.join([str(p) for p in self.paths]))
            print('\n'.join([str(Path(p) / g) for p, g in self.globs]))
            if input("Are you sure that you want to COMPLETELY delete the above items? [y/N]:").lower() != 'y':
                raise RuntimeError("Purge aborted by user")

        #remove paths
        for path in self.paths:
            path = Path(path)

            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    os.remove(path)
            except FileNotFoundError:
                print(f"Tried to purge {path} but it does not exist")
            except OSError as e:
                print(f"Tried to purge {path} but failed with OSError {e}")
        
        if len(self.globs) == 0:
            return

        #convert globs to paths in a helper purger object with force set to true
        #presumably we already have user confirmation or a force signal
        helper_purger = Purger(force=True)
        for path, glob in self.globs:
            helper_purger.add_paths(list(Path(path).glob(glob)))

        helper_purger.purge()

        self.__init__(self.force)