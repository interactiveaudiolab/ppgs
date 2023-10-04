import tarfile
import tempfile
import zipfile
from pathlib import Path

import gdown
import requests

import ppgs


###############################################################################
# Download datasets
###############################################################################


@ppgs.notify.notify_on_finish('download')
def datasets(datasets=ppgs.DATASETS, format_only=False, purge_sources=False):
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

            # Maybe clean-up
            if purge_sources:
                ppgs.data.purge.datasets(datasets=[dataset], kinds=['sources'])

        else:

            raise ValueError(f'Dataset {dataset} does not exist')


###############################################################################
# Utilities
###############################################################################


def ci_fmt(fragment):
    """Create case insensitive glob fragment"""
    characters = list(fragment.lower())
    return ''.join([f'[{c}{c.upper()}]' for c in characters])


def download_file(url, path):
    """Download file from url"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)


def download_google_drive_zip(url, path, skip_first=True):
    """Download a zip file from google drive, extract contents to path"""
    f = tempfile.NamedTemporaryFile(mode='r+b', suffix='.zip', delete=False)
    f.close()
    gdown.download(url, f.name)
    with zipfile.ZipFile(f.name) as zf:
        iterator = ppgs.iterator(
            zf.infolist()[1 if skip_first else 0:],
            'Extracting zip contents',
            total=len(zf.infolist()))
        for zipinfo in iterator:
            fname = Path(zipinfo.filename).name
            with zf.open(zipinfo, 'r') as in_file, open(path / fname, 'wb') as out_file:
                out_file.write(in_file.read())


def download_tar_bz2(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|bz2') as tstream:
            tstream.extractall(path)


def download_tar_gz(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|gz') as tstream:
            tstream.extractall(path)


def download_zip(url, path):
    """Download and extract zip file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with zipfile.ZipFile(rstream) as zstream:
            zstream.extractall(path)


def files_with_extension(ext, path):
    return list(path.rglob(f"*.{ci_fmt(ext)}"))
