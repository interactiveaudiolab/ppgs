from pathlib import Path
from typing import List, Union
from warnings import warn

import ppgs


def aggregate(
    sources: List[Union[Path, str]], 
    sinks: List[Union[Path, str]] = None,
    source_extensions: List[str] = ['wav'],
    sink_extension: str = '.pt'):
    """
    Aggregates a list of sources into a single list of files, using the provided extension to glob directories.
    """

    source_extensions = set(['.' + ext if '.' not in ext else ext for ext in source_extensions])
    sink_extension = '.' + sink_extension if '.' not in sink_extension else sink_extension

    source_files = []
    if sinks is None:
        for source in sources:
            if isinstance(source, str):
                source = Path(source)
            if not source.exists():
                raise FileNotFoundError(f'could not find source {source}')
            
            if source.is_dir():
                for extension in source_extensions:
                    source_files += list(source.rglob(f'*{extension}'))
            else:
                source_files += source

        return source_files
    else:
        sink_files = []
        for source, sink in zip(sources, sinks):
            if isinstance(source, str):
                source = Path(source)
            if isinstance(sink, str):
                sink = Path(sink)
            if not source.exists():
                raise FileNotFoundError(f'could not find source {source}')
            
            if source.is_dir():
                if not sink.is_dir():
                    raise FileNotFoundError(f'for directory source {source}, corresponding sink {sink} is not a directory')
                for extension in source_extensions:
                    source_files += list(source.rglob(f'*{extension}'))
                source_stems = [file.stem for file in source_files]
                if not len(source_stems) == len(set(source_stems)):
                    raise ValueError('two or more files have the same stem with different extensions')
                sink_files += [sink / (file.stem + sink_extension) for file in source_files]
            else:
                source_files += source
                if sink.is_dir():
                    raise OSError(f'sink {sink} is a directory')
                sink_files += sink
        return source_files, sink_files
                