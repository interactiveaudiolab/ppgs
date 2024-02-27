import csv
import re
import struct
import tarfile

import pypar

import torchutil

import ppgs


###############################################################################
# Setup TIMIT
###############################################################################


def download():
    """Prompts user to install TIMIT dataset, and formats dataset if present"""
    source_directory = ppgs.SOURCES_DIR / 'timit'
    source_directory.mkdir(parents=True, exist_ok=True)

    # Get tarball
    possible_files = [
        'timit',
        'timit.tar',
        'timit_LDC93S1.tgz',
        'timit_LDC9321.tar.gz',
        'timit.tgz',
        'timit.tar.gz']
    possible_paths = [source_directory / file for file in possible_files]
    source_exists = [path.exists() for path in possible_paths]
    try:
        chosen_source_idx = source_exists.index(True)
    except ValueError:
        raise FileNotFoundError(
            f'TIMIT dataset not found. Please download TIMIT '
            f'via https://catalog.ldc.upenn.edu/LDC93s1 '
            f'and place it in {source_directory}. '
            f'This program expects one of {possible_paths} to be present.')
    chosen_source = possible_paths[chosen_source_idx]

    # Untar
    with tarfile.open(chosen_source) as tf:
        tf.extractall(ppgs.DATA_DIR)


def format():
    """Format TIMIT"""
    data_directory = ppgs.DATA_DIR / 'timit'
    cache_directory = ppgs.CACHE_DIR / 'timit'

    # Get files
    sphere_files = ppgs.data.download.files_with_extension(
        'wav',
        data_directory)
    word_files = ppgs.data.download.files_with_extension(
        'wrd',
        data_directory)
    phone_files = ppgs.data.download.files_with_extension(
        'phn',
        data_directory)

    # Convert NIST sphere files to WAV format
    for sphere_file in torchutil.iterator(
        sphere_files,
        'Converting TIMIT audio',
        total=len(sphere_files)
    ):
        output_dir = cache_directory / sphere_file.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / (sphere_file.stem + '.wav'), 'wb') as file:
            file.write(sphere_to_wav(sphere_file))

    # Format phoneme labels
    for phone_file in torchutil.iterator(
        phone_files,
        'Converting TIMIT phonemes',
        total=len(phone_files)
    ):
        output_dir = data_directory / phone_file.parent.name / 'lab'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load phoneme alignment
        with open(phone_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
            start_times, end_times, phonemes = zip(*rows)

        # Load audio
        audio_file = (
            cache_directory /
            phone_file.parent.name /
            (phone_file.stem + '.wav'))
        audio = ppgs.load.audio(audio_file)

        # Ensure correct duration
        audio_duration = audio[0].shape[0] / ppgs.SAMPLE_RATE
        alignment_duration = float(end_times[-1]) / ppgs.SAMPLE_RATE
        if not abs(audio_duration - alignment_duration) <= 2.5e-1:
            continue

        # Write phoneme alignment
        end_times = list(end_times)
        end_times[-1] = str(audio[0].shape[0])
        rows = zip(start_times, end_times, phonemes)
        with open(output_dir / (phone_file.stem + '.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'phoneme'])
            writer.writerows(ipa_to_cmu(rows))

    # Format word alignments
    for word_file in torchutil.iterator(
        word_files,
        'Converting TIMIT word alignments',
        total=len(word_files)
    ):
        output_dir = data_directory / word_file.parent.name / 'word'
        output_dir.mkdir(parents=True, exist_ok=True)
        new_file = output_dir / (word_file.stem + '.csv')
        with open(word_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
        with open(new_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'word'])
            writer.writerows([
                [float(row[0]) / 16000, float(row[1]) / 16000, row[2]]
                for row in rows])

    # Prompt file
    prompt_file = data_directory / 'TIMIT' / 'DOC' / 'PROMPTS.TXT'
    new_file = data_directory / 'TIMIT' / 'sentences.csv'
    with open(prompt_file) as f:
        content = f.read()
    rows = [
        reversed(match) for match in re.findall(
            '(.*) \((.*)\)',
            content,
            re.MULTILINE)]
    with open(new_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prompt'])
        writer.writerows(rows)

    # Align words and phonemes
    for speaker in data_directory.iterdir():

        # Skip metadata
        if speaker.name in ['CONVERT', 'README.DOC', 'SPHERE', 'TIMIT']:
            continue

        # Align
        phone_files = list((speaker / 'lab').glob('*.csv'))
        word_dir = speaker / 'word'
        output_dir = cache_directory / speaker.name
        ppgs.data.download.align.from_files_to_files(
            phone_files,
            word_dir,
            output_dir)


###############################################################################
# Convert sphere files to wav
###############################################################################


def sphere_to_wav(sphere_file):
    """Load sphere file and convert to wav"""
    with open(sphere_file, 'rb') as f:
        header_size = sph_get_header_size(f)
        header = sph_get_header(f, header_size)
        new_header = wav_make_header(header)
        samples = sph_get_samples(f, header_size)
        return new_header + samples


###############################################################################
# Convert IPA phonemes to CMU
###############################################################################


def ipa_to_cmu(rows, backfill=True):
    """Phoneme conversion from IPA to CMU"""
    # Convert
    phones = []
    transposed = list(zip(*rows))
    for phone in transposed[2]:
        try:
            phones.append(ppgs.TIMIT_TO_ARCTIC_MAPPING[phone.lower()])
        except KeyError:
            phones.append(pypar.SILENCE)

    # Handle non-one-to-one or context-dependent phoneme conversions
    if backfill:
        backfill_indices = [
            idx for idx, phone in enumerate(phones)
            if phone[:3] == 'bck']
        for i, idx in enumerate(backfill_indices):
            assert (
                phones[idx][3] == '<' and phones[idx][-1] == '>')
            possible_replacements = phones[idx][4:-1].split(',')
            if (
                idx < len(phones) - 1 and
                phones[idx + 1] in possible_replacements
            ):
                phones[idx] = 'bck'
            else:
                phones[idx] = possible_replacements[0]
        for i in range(0, len(phones)):
            if phones[i] == 'bck':
                phones[i] = phones[i + 1]

    # Convert times to seconds
    phone_ends = [int(sample) / 16000 for sample in list(transposed[1])]

    return list(zip(phone_ends, phones))


###############################################################################
# Utilities
###############################################################################


def sph_get_header(sphere_file_object, header_size):
    """Get metadata"""
    if not hasattr(sph_get_header, 'mapping'):
        sph_get_header.mapping = {'i': int, 'r': float, 's': str}
    sphere_file_object.seek(16)
    header = sphere_file_object.read(
        header_size - 16
    ).decode('utf-8').split('\n')
    header = header[:header.index('end_head')]
    header = [
        header_item.split(' ') for header_item in header
        if header_item[0] != ';']
    return {h[0]: sph_get_header.mapping[h[1][1]](h[2]) for h in header}


def sph_get_header_size(sphere_file_object):
    """Get size of metadata in bytes"""
    sphere_file_object.seek(0)
    assert sphere_file_object.readline() == b'NIST_1A\n'
    header_size = int(sphere_file_object.readline().decode('utf-8')[:-1])
    sphere_file_object.seek(0)
    return header_size


def sph_get_samples(sphere_file_object, sphere_header_size):
    """Extract audio from sphere file"""
    sphere_file_object.seek(sphere_header_size)
    return sphere_file_object.read()


def wav_make_header(sph_header):
    """Create wav file header"""
    # Size of audio in bytes
    samples_bytes = sph_header['sample_count'] * sph_header['sample_n_bytes']

    # Create header
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        samples_bytes + 36, # total size
        b'WAVE',
        b'fmt ',
        16,  # fmt size
        1,  # header
        sph_header['channel_count'],  # channels
        sph_header['sample_rate'],  # sample rate
        sph_header['sample_rate'] * sph_header['sample_n_bytes'],  # bps
        sph_header['sample_n_bytes'],  # bytes per sample
        sph_header['sample_n_bytes']*8, # bit depth
        b'data',
        samples_bytes  # size
    )
