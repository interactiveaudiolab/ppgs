import csv
import re
import shutil

import pypar
import torchutil

import ppgs


###############################################################################
# Setup Arctic dataset
###############################################################################


def download(speakers=['bdl', 'slt', 'awb', 'jmk', 'ksp', 'clb', 'rms']):
    """Download the CMU arctic database"""
    source_directory = ppgs.SOURCES_DIR / 'arctic'
    source_directory.mkdir(parents=True, exist_ok=True)

    # Arctic has an error where one of the text files is read-only
    error_file = (
        source_directory /
        'cmu_us_ksp_arctic' /
        'etc' /
        'txt.done.data')
    error_file.unlink(missing_ok=True)

    # Download audio tarball
    for speaker in torchutil.iterator(
        speakers,
        'Downloading arctic speaker datasets',
        total=len(speakers)
    ):
        url = (
            'http://festvox.org/cmu_arctic/cmu_arctic/'
            f'packed/cmu_us_{speaker}_arctic-0.95-release.tar.bz2')
        torchutil.download.tarbz2(url, source_directory)

    # Download metadata
    torchutil.download.file(
        'http://festvox.org/cmu_arctic/cmuarctic.data',
        source_directory / 'sentences.txt')


def format(speakers=None):
    """Formats the CMU Arctic database"""
    source_directory = ppgs.SOURCES_DIR / 'arctic'
    data_directory = ppgs.DATA_DIR / 'arctic'
    cache_directory = ppgs.CACHE_DIR / 'arctic'
    data_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    # Copy text and alignments
    sentences_file = source_directory / 'sentences.txt'
    new_sentences_file = data_directory / 'sentences.csv'
    with open(sentences_file, 'r') as file:
        content = file.read()
    rows = [
        match for match in re.findall(
            r'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"(.+)\" \)',
            content,
            re.MULTILINE)]
    with open(new_sentences_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['id','prompt'])
        writer.writerows(rows)

    # Get arctic speakers
    if speakers is None:
        speakers = list(source_directory.glob('cmu_us_*_arctic'))
    else:
        speakers = [
            source_directory / f'cmu_us_{speaker}_arctic'
            for speaker in speakers]

    # Format each speaker
    for speaker in torchutil.iterator(
        speakers,
        'Formatting arctic speakers',
        total=len(speakers)
    ):

        # Map version 0.90 ids to version 0.95 ids
        if speaker.name == 'cmu_us_awb_arctic':
            v90 = speaker / 'etc' / 'txt.done.data'
            v95 = sentences_file
            with open(v90) as f:
                cv90 = f.read()
            with open(v95) as f:
                cv95 = f.read()
            id_map = lambda id: version_90_to_version_95(id, cv90, cv95)
        else:
            id_map = lambda id: id

        new_speaker_dir = data_directory / speaker.name
        cache_speaker_dir = cache_directory / speaker.name
        cache_speaker_dir.mkdir(parents=True, exist_ok=True)
        lab_dir_path = speaker / 'lab'
        wav_dir_path = speaker / 'wav'
        new_lab_dir_path = new_speaker_dir / 'lab'
        new_lab_dir_path.mkdir(parents=True, exist_ok=True)

        # Get label files
        lab_files = ppgs.data.download.files_with_extension(
            'lab',
            lab_dir_path)

        new_phone_files = []
        for lab_file in lab_files:

            # Necessary for weird extra file included in some arctic versions
            if lab_file.stem == '*':
                continue

            # Read file
            with open(lab_file, 'r') as f:
                lines = f.readlines()

            # Remove header
            non_header_lines = lines[lines.index('#\n') + 1:]

            # Parse phoneme alignment
            timestamps, _, phonemes = zip(*[
                line.split() for line in non_header_lines if len(line) >= 5])

            # Map unknown tokens to silence
            phonemes = [
                pypar.SILENCE if phone not in ppgs.PHONEMES else phone
                for phone in phonemes]

            # Load audio
            audio = ppgs.load.audio(wav_dir_path / (lab_file.stem + '.wav'))

            # Skip if length disagrees
            audio_duration = audio[0].shape[0] / ppgs.SAMPLE_RATE
            if abs(audio_duration - float(timestamps[-1])) > 1e-1:
                continue

            # Skip utterances with unknown correspondence
            stem = id_map(lab_file.stem)
            if stem is None:
                continue

            # Write alignment to CSV
            timestamps = list(timestamps)
            timestamps[-1] = str(audio_duration)
            rows = zip(timestamps, phonemes)
            new_phone_file = new_lab_dir_path / f'{stem}.csv'
            new_phone_files.append(new_phone_file)
            with open(new_phone_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'phoneme'])
                writer.writerows(rows)

        # Transfer audio files
        wav_files = ppgs.data.download.files_with_extension(
            'wav',
            wav_dir_path)
        for wav_file in wav_files:
            try:
                shutil.copy(
                    wav_file,
                    cache_speaker_dir / (id_map(wav_file.stem) + '.wav'))
            except TypeError:
                continue

        # Align phonemes with words
        new_word_dir = new_speaker_dir / 'word'
        new_word_dir.mkdir(parents=True, exist_ok=True)
        new_word_files = [
            new_word_dir / (file.stem + '.csv') for file in new_phone_files]
        ppgs.data.datasets.arctic.words.from_files_to_files(
            new_phone_files,
            new_word_files,
            new_sentences_file)
        ppgs.data.download.align.from_files_to_files(
            new_phone_files,
            new_word_dir,
            cache_speaker_dir)


###############################################################################
# Utilities
###############################################################################


def version_90_to_version_95(id, v90_sentences, v95_sentences):
    """Maps Arctic sentence ids from version 0.90 to version 0.95"""
    # Get sentence id
    sentence = re.search(rf'\( {id} \"(.+)\" \)', v90_sentences).groups()[0]

    try:

        # Find matching sentence id
        return re.search(
            rf'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"{sentence}\" \)',
            v95_sentences
        ).groups()[0]

    except AttributeError:

        # No match
        return None
