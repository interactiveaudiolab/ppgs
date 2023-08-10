import ppgs
from ppgs.data.download.utils import *
from . import words
from ppgs.data.download import align
from .version import v0_90_to_v0_95
import pypar
from shutil import copy as cp
import csv
import re
import tqdm

def download(arctic_speakers=['bdl', 'slt', 'awb', 'jmk', 'ksp', 'clb', 'rms']):
    """Downloads the CMU arctic database"""
    arctic_sources = ppgs.SOURCES_DIR / 'arctic'
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

def format(speakers=None):
    """Formats the CMU Arctic database"""

    arctic_sources = ppgs.SOURCES_DIR / 'arctic'
    arctic_data = ppgs.DATA_DIR / 'arctic'
    if not arctic_sources.exists():
        raise FileNotFoundError(f"'{arctic_sources}' does not exist")
    if not arctic_data.exists():
        arctic_data.mkdir(parents=True, exist_ok=True)
    cache_dir = ppgs.CACHE_DIR / 'arctic'

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

    #get arctic speakers
    speakers = list(arctic_sources.glob('cmu_us_*_arctic')) if speakers is None \
        else [arctic_sources / f"cmu_us_{speaker}_arctic" for speaker in speakers]

    iterator = tqdm.tqdm(
        speakers,
        desc='Formatting arctic speakers',
        total = len(speakers),
        dynamic_ncols=True
    )
    #iterate speakers and copy
    for speaker in iterator:
        if speaker.name == 'cmu_us_awb_arctic': #map version 0.90 ids to version 0.95 ids
            v90 = speaker / 'etc' / 'txt.done.data'
            v95 = sentences_file
            with open(v90) as f:
                cv90 = f.read()
            with open(v95) as f:
                cv95 = f.read()
            id_map = lambda id: v0_90_to_v0_95(id, cv90, cv95)
        else:
            id_map = lambda id: id
        new_speaker_dir = arctic_data / speaker.name
        cache_speaker_dir = cache_dir / speaker.name
        cache_speaker_dir.mkdir(parents=True, exist_ok=True)

        #transfer phoneme label files
        lab_dir_path = speaker / 'lab'
        wav_dir_path = speaker / 'wav'
        new_lab_dir_path = new_speaker_dir / 'lab'

        if not lab_dir_path.exists():
            raise ValueError(f'could not find directory {lab_dir_path}')

        #create destination directory
        new_lab_dir_path.mkdir(parents=True, exist_ok=True)

        #get label files
        lab_files = files_with_extension('lab', lab_dir_path)
        new_phone_files = []

        nested_iterator = tqdm.tqdm(
            lab_files,
            desc=f'transferring phonetic label files for arctic speaker {speaker.name}',
            total = len(lab_files),
            dynamic_ncols=True
        )
        for lab_file in nested_iterator:
            if lab_file.stem == '*': #necessary for weird extra file included in some arctic versions
                continue
            with open(lab_file, 'r') as f:
                lines = f.readlines()
                non_header_lines = lines[lines.index('#\n')+1:] #get rid of useless headers
                timestamps, _, phonemes = zip(*[line.split() for line in non_header_lines if len(line) >= 5])
                #Map special case of silence map from pau to SILENCE
                phonemes = [pypar.SILENCE if phone == 'pau' else phone for phone in phonemes]
                #Map errors to <unk>
                phonemes = [phone if phone in ppgs.PHONEME_LIST else pypar.SILENCE for phone in phonemes]
            # with open(wav_dir_path / (lab_file.stem + '.wav'), 'rb') as f:
                # audio = ppgs.load.audio(f)
                audio = ppgs.load.audio(wav_dir_path / (lab_file.stem + '.wav'))
                audio_duration = audio[0].shape[0] / ppgs.SAMPLE_RATE
                if not abs(audio_duration - float(timestamps[-1])) <= 1e-1:
                    print(f'failed with stem {lab_file.stem}')
                    continue
                timestamps = list(timestamps)
                timestamps[-1] = str(audio_duration)
            rows = zip(timestamps, phonemes)
            #write new label file as CSV
            try:
                new_phone_file = new_lab_dir_path / (id_map(lab_file.stem) + '.csv')
            except TypeError:
                continue
            new_phone_files.append(new_phone_file)
            with open(new_phone_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'phoneme'])
                writer.writerows(rows)

        #transfer wav files
        import pdb; pdb.set_trace()
        new_wav_dir_path = cache_speaker_dir
        new_wav_dir_path.mkdir(parents=True, exist_ok=True)
        if not wav_dir_path.exists():
            raise FileNotFoundError(f'could not find directory {wav_dir_path}')

        wav_files = files_with_extension('wav', wav_dir_path)

        nested_iterator = tqdm.tqdm(
            wav_files,
            desc=f'Transferring audio files for arctic speaker {speaker.name}',
            total=len(wav_files),
            dynamic_ncols=True
        )
        for wav_file in nested_iterator:
            try:
                cp(wav_file, new_wav_dir_path / (id_map(wav_file.stem) + '.wav'))
            except TypeError:
                continue

        #create word alignment files
        new_word_dir = new_speaker_dir / 'word'
        new_word_files = [new_word_dir / (file.stem + '.csv') for file in new_phone_files]

        if not new_word_dir.exists():
            new_word_dir.mkdir(parents=True, exist_ok=True)

        words.from_files_to_files(new_phone_files, new_word_files, new_sentences_file)

        align.from_files_to_files(new_phone_files, new_word_dir, cache_speaker_dir)