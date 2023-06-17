import ppgs
from pathlib import Path
from typing import List, Union
from warnings import warn

def aggregate(sources: List[Union[Path, str]], extensions: List[str] = []):
    """
    Aggregates a list of sources into a single list of files, using the provided extension to glob directories.
    """

    extensions = ['.' + ext if '.' not in ext else ext for ext in extensions]

    files = []
    for source in sources:
        if isinstance(source, str):
            source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f'could not find source {source}')
        
        if source.is_dir():
            for extension in extensions:
                files += list(source.rglob(f'*{extension}'))
        else:
            files += source

    return files