import requests
import tarfile

def download_file(url, file):
    """Download file from url"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(file, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)

def download_tar_bz2(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode="r|bz2") as tstream:
            tstream.extractall(path)

def ci_fmt(fragment):
    """Create case insensitive glob fragment"""
    characters = list(fragment.lower())
    return ''.join([f'[{c}{c.upper()}]' for c in characters])

def files_with_extension(ext, path):
    return list(path.rglob(f"*.{ci_fmt(ext)}"))