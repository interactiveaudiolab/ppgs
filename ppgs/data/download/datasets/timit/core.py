import ppgs
from ppgs.data.download.utils import *
from .phones import timit_to_arctic
from .sph import pcm_sph_to_wav
from pathlib import Path
import tarfile
import csv
import re

def download(timit_source=None):
    """Prompts user to install TIMIT dataset, and formats dataset if present"""

    ppgs.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    if timit_source is None:
        possible_files = [ #in order of preferrence
            'timit',
            'timit.tar',
            'timit_LDC93S1.tgz',
            'timit_LDC9321.tar.gz',
            'timit.tgz',
            'timit.tar.gz'
        ]
        possible_paths = [ppgs.SOURCES_DIR / file for file in possible_files]
        source_exists = [path.exists() for path in possible_paths]
        try:
            chosen_source_idx = source_exists.index(True)
        except ValueError:
            raise FileNotFoundError(f"""TIMIT dataset can only be officially downloaded via https://catalog.ldc.upenn.edu/LDC93s1,
            please download this resource and place it in '{ppgs.SOURCES_DIR}'. This command expects one of {possible_paths} to be present""")
        chosen_source = possible_paths[chosen_source_idx]
    else:
        timit_source = Path(timit_source)
        if not timit_source.exists():
            raise FileNotFoundError(f"User specified file {timit_source} does not exist")
        chosen_source = timit_source
    print(f"Using '{chosen_source}' as source for TIMIT dataset")
    if chosen_source_idx > 0:
        print(f"unzipping {chosen_source} to '{ppgs.SOURCES_DIR}'")
        with tarfile.open(chosen_source) as tf:
            tf.extractall(ppgs.SOURCES_DIR)
            if not (ppgs.SOURCES_DIR / 'timit').exists():
                raise FileNotFoundError(f"'{ppgs.SOURCES_DIR}/timit' should exist now, but it does not")
        download(timit_source)

def format_timit():
    """Formats the TIMIT database"""

    #walk filetree and find files
    timit_sources = ppgs.SOURCES_DIR / 'timit/TIMIT'
    timit_data = ppgs.DATA_DIR / 'timit'
    if not timit_sources.exists():
        raise FileNotFoundError(f"'{timit_sources}' does not exist")
    sphere_files = files_with_extension('wav', timit_sources)
    word_files = files_with_extension('wrd', timit_sources)
    phone_files = files_with_extension('phn', timit_sources)

    #convert NIST sphere files to WAV format and transfer
    iterator = tqdm.tqdm(
        sphere_files,
        desc='Converting NIST sphere to WAV format',
        total=len(sphere_files),
        dynamic_ncols=True
    )
    for sphere_file in iterator:
        output_dir = timit_data / sphere_file.parent.name / 'wav'
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / (sphere_file.stem + '.wav')
        if not new_path.exists():
            with open(new_path, 'wb') as f:
                f.write(pcm_sph_to_wav(sphere_file))


    #convert and transfer phoneme label files
    iterator = tqdm.tqdm(
        phone_files,
        desc='Converting phonetic label files for TIMIT dataset',
        total=len(phone_files),
        dynamic_ncols=True
    )
    for phone_file in iterator:
        output_dir = timit_data / phone_file.parent.name / 'lab'
        output_dir.mkdir(parents=True, exist_ok=True)
        new_file = output_dir / (phone_file.stem + '.csv')
        corresponding_audio_file = output_dir.parents[0] / 'wav' / (phone_file.stem + '.wav')
        with open(phone_file, 'r') as f: #Get phone file contents
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
            start_times, end_times, phonemes = zip(*rows)
        audio = ppgs.load.audio(corresponding_audio_file)
        audio_duration = audio[0].shape[0] / ppgs.SAMPLE_RATE
        if not abs(audio_duration - (float(end_times[-1])/ppgs.SAMPLE_RATE)) <= 2.5e-1:
            print(f'failed with stem {phone_file.stem}')
            continue
        end_times = list(end_times)
        end_times[-1] = str(audio[0].shape[0])
        rows = zip(start_times, end_times, phonemes)
        with open(new_file, 'w') as f: #Write phone csv file
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'phoneme'])
            writer.writerows(timit_to_arctic(rows))

    #Word timing files
    iterator = tqdm.tqdm(
        word_files,
        desc='Converting and transfering word timing files for TIMIT dataset',
        total=len(word_files),
        dynamic_ncols=True
    )
    for word_file in iterator:
        output_dir = timit_data / word_file.parent.name / 'word'
        output_dir.mkdir(parents=True, exist_ok=True)
        new_file = output_dir / (word_file.stem + '.csv')
        with open(word_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
        with open(new_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['start', 'end', 'word'])
            writer.writerows([[float(row[0])/16000, float(row[1])/16000, row[2]] for row in rows])

    #Prompt file
    prompt_file = timit_sources / 'DOC' / 'PROMPTS.TXT'
    new_file = timit_data / 'sentences.csv'
    with open(prompt_file) as f:
        content = f.read()
    rows = [reversed(match) for match in re.findall('(.*) \((.*)\)', content, re.MULTILINE)]
    with open(new_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prompt'])
        writer.writerows(rows)