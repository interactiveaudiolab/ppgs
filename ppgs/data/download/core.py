import tempfile
import zipfile
from pathlib import Path

import torchutil

import ppgs


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets=ppgs.DATASETS, format_only=False):
    """Downloads datasets"""
    for dataset in [dataset.lower() for dataset in datasets]:

        if hasattr(ppgs.data.datasets, dataset):

            # Get dataset submodule
            dataset_object = getattr(ppgs.data.datasets, dataset)

            # Download
            if not format_only:
                dataset_object.download()

            # Extract relevant data in common format
            dataset_object.format()

        else:

            raise ValueError(f'Dataset {dataset} does not exist')


###############################################################################
# Utilities
###############################################################################


def ci_fmt(fragment):
    """Create case insensitive glob fragment"""
    characters = list(fragment.lower())
    return ''.join([f'[{c}{c.upper()}]' for c in characters])


def download_google_drive_zip(url, path, skip_first=True):
    """Download a zip file from google drive, extract contents to path"""
    import gdown
    f = tempfile.NamedTemporaryFile(mode='r+b', suffix='.zip', delete=False)
    f.close()
    gdown.download(url, f.name)
    with zipfile.ZipFile(f.name) as zf:
        for zipinfo in torchutil.iterator(
            zf.infolist()[1 if skip_first else 0:],
            'Extracting zip contents',
            total=len(zf.infolist())
        ):
            fname = Path(zipinfo.filename).name
            with zf.open(zipinfo, 'r') as in_file, open(path / fname, 'wb') as out_file:
                out_file.write(in_file.read())


def files_with_extension(ext, path):
    return list(path.rglob(f"*.{ci_fmt(ext)}"))
