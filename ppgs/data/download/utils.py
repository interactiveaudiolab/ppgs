import requests
import tarfile
from zipfile import ZipFile
import gdown
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from pathlib import Path

def download_file(url, path):
    """Download file from url"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)

def download_google_drive_zip(url, path, skip_first=True):
    """Download a zip file from google drive, extract contents to path"""
    f = NamedTemporaryFile(mode='r+b', suffix=".zip", delete=False)
    f.close()
    gdown.download(url, f.name)
    with open(f.name, 'r+b') as f:
        with ZipFile(f) as zf:
            iterator = tqdm(
                zf.infolist()[1 if skip_first else 0:],
                desc="Extracting zip contents",
                total=len(zf.infolist()),
                dynamic_ncols=True
            )
            for zipinfo in iterator:
                fname = Path(zipinfo.filename).name
                with zf.open(zipinfo, 'r') as in_file, open(path / fname, 'wb') as out_file:
                    out_file.write(in_file.read())
    #TODO delete temp file

def download_zip(url, path):
    """Download and extract zip file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with ZipFile(rstream) as zstream:
            zstream.extractall(path)

def download_tar_bz2(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode="r|bz2") as tstream:
            tstream.extractall(path)

def download_tar_gz(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode="r|gz") as tstream:
            tstream.extractall(path)

def ci_fmt(fragment):
    """Create case insensitive glob fragment"""
    characters = list(fragment.lower())
    return ''.join([f'[{c}{c.upper()}]' for c in characters])

def files_with_extension(ext, path):
    return list(path.rglob(f"*.{ci_fmt(ext)}"))