import tarfile
from shutil import copy as cp
import csv
import tqdm
from pathlib import Path
import re

from ppgs import SOURCES_DIR, DATA_DIR
#DEBUG
SOURCES_DIR = Path('/home/cameron/ppgs/tmp/data/sources/')
DATA_DIR = Path('/home/cameron/ppgs/tmp/data/datasets/')

from .sph import pcm_sph_to_wav
from .utils import files_with_extension, download_file, download_tar_bz2
from .phones import timit_to_arctic

def datasets(datasets, format_only, timit_source, arctic_speakers):
    """Downloads the datasets passed in"""
    datasets = [dataset.lower() for dataset in datasets]
    if 'timit' in datasets:
        if not format_only:
            download_timit(timit_source)
            format_timit()
        else:
            format_timit()
    if 'arctic' in datasets:
        if not format_only:
            download_arctic(arctic_speakers)
            format_arctic()
        else:
            format_arctic()

###############################################################################
# Downloading
###############################################################################

def download_timit(timit_source=None):
    """Prompts user to install TIMIT dataset, and formats dataset if present"""
    
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    if timit_source is None:
        possible_files = [ #in order of preferrence
            'timit',
            'timit.tar',
            'timit_LDC93S1.tgz',
            'timit_LDC9321.tar.gz',
            'timit.tgz',
            'timit.tar.gz'
        ]
        possible_paths = [SOURCES_DIR / file for file in possible_files]
        source_exists = [path.exists() for path in possible_paths]
        try:
            chosen_source_idx = source_exists.index(True)
        except ValueError:
            raise FileNotFoundError(f"""TIMIT dataset can only be officially downloaded via https://catalog.ldc.upenn.edu/LDC93s1,
            please download this resource and place it in '{SOURCES_DIR}'. This command expects one of {possible_paths} to be present
            or a user provided path using '--timit-source' argument""")
        chosen_source = possible_paths[chosen_source_idx]
    else:
        timit_source = Path(timit_source)
        if not timit_source.exists():
            raise FileNotFoundError(f"User specified file {timit_source} does not exist")
        chosen_source = timit_source
    print(f"Using '{chosen_source}' as source for TIMIT dataset")
    if chosen_source_idx > 0:
        print(f"unzipping {chosen_source} to '{SOURCES_DIR}'")
        with tarfile.open(chosen_source) as tf:
            #TODO technically unsafe if infected timit source is used (low probability event)
            tf.extractall(SOURCES_DIR)
            if not (SOURCES_DIR / 'timit').exists():
                raise FileNotFoundError(f"'{SOURCES_DIR}/timit' should exist now, but it does not")
        download_timit(timit_source)

def download_arctic(arctic_speakers):
    """Downloads the CMU arctic database"""
    arctic_sources = SOURCES_DIR / 'arctic'
    arctic_sources.mkdir(parents=True, exist_ok=True)
    iterator = tqdm.tqdm(
        arctic_speakers,
        desc='Downloading arctic speaker datasets',
        total=len(arctic_speakers),
        dynamic_ncols=True
    )
    for arctic_speaker in iterator:
        if not (arctic_sources / f"cmu_us_{arctic_speaker}_arctic").exists():
            url = f"http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_{arctic_speaker}_arctic-0.95-release.tar.bz2"
            download_tar_bz2(url, arctic_sources)
    download_file('http://festvox.org/cmu_arctic/cmuarctic.data', arctic_sources / 'sentences.txt')


###############################################################################
# Formatting
###############################################################################

def format_timit():
    """Formats the TIMIT database"""

    #walk filetree and find files
    timit_sources = SOURCES_DIR / 'timit/TIMIT'
    timit_data = DATA_DIR / 'timit'
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
        with open(phone_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
        with open(new_file, 'w') as f:
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
            writer.writerows(rows)

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
    


def format_arctic():
    """Formats the CMU Arctic database"""

    arctic_sources = SOURCES_DIR / 'arctic'
    arctic_data = DATA_DIR / 'arctic'
    if not arctic_sources.exists():
        raise FileNotFoundError(f"'{arctic_sources}' does not exist")

    #get arctic speakers
    speakers = list(arctic_sources.glob('cmu_us_*_arctic'))

    iterator = tqdm.tqdm(
        speakers,
        desc='Formatting arctic speakers',
        total = len(speakers),
        dynamic_ncols=True
    )

    #iterate speakers and copy
    for speaker in iterator:
        new_speaker_dir = arctic_data / speaker.name

        #transfer phoneme label files
        lab_dir_path = speaker / 'lab'
        new_lab_dir_path = new_speaker_dir / 'lab'

        if not lab_dir_path.exists():
            raise ValueError(f'could not find directory {lab_dir_path}')

        #create destination directory
        new_lab_dir_path.mkdir(parents=True, exist_ok=True)

        #get label files
        lab_files = files_with_extension('lab', lab_dir_path)

        nested_iterator = tqdm.tqdm(
            lab_files,
            desc=f'transferring phonetic label files for arctic speaker {speaker.name}',
            total = len(lab_files),
            dynamic_ncols=True
        )
        for lab_file in nested_iterator:
            if lab_file.stem == '*':
                continue
            with open(lab_file, 'r') as f:
                lines = f.readlines()
                non_header_lines = lines[lines.index('#\n')+1:] #get rid of useless headers
                timestamps, _, phonemes = zip(*[line.split() for line in non_header_lines if len(line) >= 5])
                rows = zip(timestamps, phonemes)
            #write new label file as CSV
            with open(new_lab_dir_path / (lab_file.stem + '.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'phoneme'])
                writer.writerows(rows)

        #transfer wav files
        wav_dir_path = speaker / 'wav'
        new_wav_dir_path = new_speaker_dir / 'wav'

        if not wav_dir_path.exists():
            raise FileNotFoundError(f'could not find directory {wav_dir_path}')

        new_wav_dir_path.mkdir(parents=True, exist_ok=True)
        wav_files = files_with_extension('wav', wav_dir_path)

        nested_iterator = tqdm.tqdm(
            wav_files,
            desc=f'Transferring audio files for arctic speaker {speaker.name}',
            total=len(wav_files),
            dynamic_ncols=True
        )
        for wav_file in nested_iterator:
            cp(wav_file, new_wav_dir_path / (wav_file.stem + '.wav'))

    #transfer sentences file
    sentences_file = arctic_sources / 'sentences.txt'
    new_sentences_file = arctic_data / 'sentences.csv'
    if not sentences_file.exists():
        raise FileNotFoundError(f'could not find sentences file {sentences_file}')
    with open(sentences_file, 'r') as f:
        content = f.read()
    rows = [match for match in re.findall(r'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"(.+)\" \)', content, re.MULTILINE)]
    with open(new_sentences_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','prompt'])
        writer.writerows(rows)
    