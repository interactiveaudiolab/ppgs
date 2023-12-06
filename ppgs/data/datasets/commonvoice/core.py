import functools
import tarfile
from pathlib import Path

import pypar
import torchaudio
import torchutil

import ppgs


###############################################################################
# Setup Common Voice
###############################################################################


def download():
    """Downloads the Charsiu-aligned subset of Common Voice"""
    source_directory = ppgs.SOURCES_DIR / 'commonvoice'
    source_directory.mkdir(parents=True, exist_ok=True)
    data_directory = ppgs.DATA_DIR / 'commonvoice'
    data_directory.mkdir(parents=True, exist_ok=True)

    # Download alignments
    alignment_directory = source_directory / 'alignments'
    alignment_directory.mkdir(parents=True, exist_ok=True)
    ppgs.data.download.download_google_drive_zip(
        'https://drive.google.com/uc?id=1J_IN8HWPXaKVYHaAf7IXzUd6wyiL9VpP',
        alignment_directory)

    # Untar audio
    mp3_dir = data_directory / 'mp3'
    mp3_dir.mkdir(exist_ok=True)
    cv_corpus_files = (
        list(source_directory.glob('cv-corpus*.tar.gz')) +
        list(source_directory.glob('cv-corpus*.tgz')))
    if cv_corpus_files:
        corpus_file = sorted(cv_corpus_files)[-1]
        stems = set([
            file.stem for file in ppgs.data.download.files_with_extension(
                'TextGrid',
                alignment_directory)])
        with tarfile.open(corpus_file, 'r|gz') as corpus:
            for file_info in corpus:
                stem = Path(file_info.name).stem
                if stem in stems:
                    stems.discard(stem)
                    file_contents = corpus.extractfile(file_info).read()
                    with open(mp3_dir / (stem + '.mp3'), 'wb') as f:
                        f.write(file_contents)
    else:
        raise FileNotFoundError(
            f'The Common Voice Dataset can only be officially downloaded '
            f'via https://commonvoice.mozilla.org/en, please download this '
            f'resource and place it in {source_directory}. This program '
            f'expects Common Voice tar.gz or tgz files to be present.')


def format():
    """Formats the Common Voice dataset"""
    source_directory = ppgs.SOURCES_DIR / 'commonvoice'

    # Get alignment files
    textgrid_files = ppgs.data.download.files_with_extension(
        'TextGrid',
        source_directory)
    stems = set([f.stem for f in textgrid_files])

    # Get audio files
    mp3_files = list(
        ppgs.data.download.files_with_extension(
            'mp3',
            ppgs.DATA_DIR / 'commonvoice' / 'mp3'))

    # Get correspondence between audio and alignment files
    found_stems = []
    mp3_found = []
    for mp3_file in mp3_files:
        if mp3_file.stem in stems:
            found_stems.append(mp3_file.stem)
            mp3_found.append(mp3_file)

    # Multiprocessed formatting
    cache_directory = ppgs.CACHE_DIR / 'commonvoice'
    cache_directory.mkdir(exist_ok=True, parents=True)
    process = functools.partial(
        mp3_textgrid,
        audio_directory=cache_directory,
        alignment_directory=cache_directory,
        source_directory=source_directory)
    torchutil.multiprocess_iterator(
        process,
        mp3_found,
        message='Formatting Common Voice',
        num_workers=ppgs.NUM_WORKERS,
        worker_chunk_size=512)


###############################################################################
# Utilities
###############################################################################


def mp3_textgrid(
    mp3_file: Path,
    audio_directory=None,
    alignment_directory=None,
    source_directory=None):
    """Worker for handling mp3 and textgrid conversion"""
    # Resample and save audio
    audio = ppgs.load.audio(mp3_file)
    torchaudio.save(
        audio_directory / (mp3_file.stem + '.wav'),
        audio,
        sample_rate=ppgs.SAMPLE_RATE)
    duration = audio.shape[-1] / ppgs.SAMPLE_RATE

    # Load alignment file contents
    textgrid_file = (
        source_directory /
        'alignments' /
        (mp3_file.stem + '.TextGrid'))
    with open(textgrid_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Replace header
    lines[0] = 'File type = "ooTextFile short"\n'
    lines[1] = '"TextGrid"\n'

    # Save alignment file
    output_textgrid = alignment_directory / textgrid_file.name
    with open(output_textgrid, 'w', encoding='utf-8') as outfile:
        outfile.writelines(lines)

    # Load alignment
    alignment = pypar.Alignment(output_textgrid)

    # Replace silence tokens
    for i in range(len(alignment)):
        if str(alignment[i]) == '[SIL]':
            alignment[i].word = pypar.SILENCE
        for j in range(len(alignment[i])):
            if str(alignment[i][j]) == '[SIL]':
                alignment[i][j].phoneme = pypar.SILENCE

    # Align end time of final phoneme with audio duration
    alignment[-1][-1]._end = duration

    # Save alignment
    alignment.save(output_textgrid)
