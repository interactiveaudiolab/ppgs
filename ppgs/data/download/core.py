import tarfile
from shutil import copy as cp
import csv
import tqdm
from pathlib import Path
import re
import torchaudio

import ppgs
from ppgs import SOURCES_DIR, DATA_DIR
import ppgs.data.purge

from .sph import pcm_sph_to_wav
from .utils import files_with_extension, download_file, download_tar_bz2, download_google_drive_zip, download_tar_gz
from .phones import timit_to_arctic
from .arctic_version import v0_90_to_v0_95

#TODO this file could use a refactor

def datasets(datasets, format_only, timit_source, common_voice_source, arctic_speakers, purge_sources):
    """Downloads the datasets passed in"""
    datasets = [dataset.lower() for dataset in datasets]
    if 'timit' in datasets:
        if not format_only:
            download_timit(timit_source)
            format_timit()
        else:
            format_timit()
        if purge_sources:
            ppgs.data.purge.datasets(datasets=['timit'], kinds=['sources'])
    if 'arctic' in datasets:
        if not format_only:
            download_arctic(arctic_speakers)
            format_arctic(arctic_speakers)
        else:
            format_arctic(arctic_speakers)
        if purge_sources:
            ppgs.data.purge.datasets(datasets=['arctic'], kinds=['sources'])
    if 'charsiu' in datasets:
        if not format_only:
            download_charsiu(common_voice_source)
            format_charsiu()
        else:
            format_charsiu()
        if purge_sources:
            ppgs.data.purge.datasets(datasets=['charsiu'], kinds=['sources'])


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

def download_charsiu(common_voice_source=None):
    """Downloads the Charsiu MFA aligned dataset, which includes a subset of Common Voice"""
    charsiu_sources = SOURCES_DIR / 'charsiu'
    charsiu_sources.mkdir(parents=True, exist_ok=True)

    #download TextGrid files
    alignments_dir = charsiu_sources / 'alignments'
    alignments_dir.mkdir(parents=True, exist_ok=True)
    download_google_drive_zip('https://drive.google.com/uc?id=1J_IN8HWPXaKVYHaAf7IXzUd6wyiL9VpP', alignments_dir)

    #download Common Voice Subset
    if common_voice_source is None:
        #TODO make this work with directories (who would ever want that?)
        cv_corpus_files = list(SOURCES_DIR.glob('cv-corpus*.tar.gz')) + list(SOURCES_DIR.glob('cv-corpus*.tgz'))
        if len(cv_corpus_files) >= 1:
            corpus_file = sorted(cv_corpus_files)[-1]
            # stems = [file.stem for file in files_with_extension('textgrid', alignments_dir)]
            corpus = tarfile.open(corpus_file, 'r|gz')
            common_voice_dir = charsiu_sources / 'common_voices'
            corpus.extractall(path=common_voice_dir)
            # base_path = Path(list(Path(corpus.next().path).parents)[-2]) #get base directory of tarfile
            # clips_path = base_path / 'en' / 'clips' #TODO make language configurable?
            # mp3_dir = charsiu_sources / 'mp3'
            # mp3_dir.mkdir(exist_ok=True, parents=True)
            # print('Scanning tar contents, this can take a long time (>10 minutes)')
            # contents = corpus.getnames()
            # iterator = tqdm.tqdm(
            #     stems,
            #     desc="Extracting common voice clips with corresponding Charsiu alignments",
            #     total=len(stems),
            #     dynamic_ncols=True
            # )
            # stems_not_found = []
            # for stem in iterator:
            #     mp3_path = str(clips_path / (stem + '.mp3'))
            #     mp3_info = corpus.getmember(mp3_path)
            #     try:
            #         mp3_file = corpus.extractfile(mp3_info)
            #         with open(mp3_dir / (stem + '.mp3'), 'wb') as new_mp3_file:
            #             import pdb; pdb.set_trace()
            #             new_mp3_file.write(mp3_file.read())
            #         mp3_file.close()
            #     except:
            #         stems_not_found.append(stem)

        else:
            raise FileNotFoundError(f"""The Common Voice Dataset can only be officially downloaded via https://commonvoice.mozilla.org/en,
            please download this resource and place it in '{SOURCES_DIR}'. This command expects a tar.gz or tgz to be present
            or a user provided path using '--common-voice-source' argument""")
    

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
        corresponding_audio_file = output_dir.parents[0] / 'wav' / (phone_file.stem + '.wav')
        with open(phone_file, 'r') as f: #Get phone file contents
            reader = csv.reader(f, delimiter=' ')
            rows = list(reader)
            start_times, end_times, phonemes = zip(*rows)
        with open(corresponding_audio_file, 'rb') as f:
            audio = ppgs.load.audio(f)
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
    


def format_arctic(speakers=None):
    """Formats the CMU Arctic database"""

    arctic_sources = SOURCES_DIR / 'arctic'
    arctic_data = DATA_DIR / 'arctic'
    if not arctic_sources.exists():
        raise FileNotFoundError(f"'{arctic_sources}' does not exist")
    if not arctic_data.exists():
        arctic_data.mkdir(parents=True, exist_ok=True)

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
                #Map special case of silence map from pau to sp
                phonemes = ['sp' if phone == 'pau' else phone for phone in phonemes]
                #Map errors to <unk>
                phonemes = [phone if phone in ppgs.PHONEME_LIST else '<unk>' for phone in phonemes]
            with open(wav_dir_path / (lab_file.stem + '.wav'), 'rb') as f:
                audio = ppgs.load.audio(f)
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
            try:
                cp(wav_file, new_wav_dir_path / (id_map(wav_file.stem) + '.wav'))
            except TypeError:
                continue

        #create word alignment files
        new_word_dir = new_speaker_dir / 'word'
        new_word_files = [new_word_dir / (file.stem + '.csv') for file in new_phone_files]

        if not new_word_dir.exists():
            new_word_dir.mkdir(parents=True, exist_ok=True)

        ppgs.data.download.words.from_files_to_files(new_phone_files, new_word_files, new_sentences_file)
    

def format_charsiu():
    """Formats the charsiu dataset"""
    charsiu_sources = SOURCES_DIR / 'charsiu'
    
    print("Scanning charsiu for textgrid files (operation may be slow)")
    textgrid_files = files_with_extension('textgrid', charsiu_sources)

    stems = set([f.stem for f in textgrid_files])

    mp3_files = list(files_with_extension('mp3', charsiu_sources))

    found_stems = []
    mp3_found = []
    iterator = tqdm.tqdm(
        mp3_files,
        desc="Scanning Common Voice for mp3 files matching charsiu textgrid labels",
        total=len(mp3_files),
        dynamic_ncols=True
    )

    for mp3_file in iterator:
        if mp3_file.stem in stems:
            found_stems.append(mp3_file.stem)
            mp3_found.append(mp3_file)

    num_not_found = len(stems.difference(set(found_stems)))
    
    print(f"Failed to find {num_not_found}/{len(stems)} mp3 files ({num_not_found/(len(stems)+1e-6)*100}%)!")

    charsiu_data_dir = ppgs.DATA_DIR / 'charsiu'
    charsiu_wav_dir = charsiu_data_dir / 'wav'
    charsiu_textgrid_dir = charsiu_data_dir / 'textgrid'

    charsiu_wav_dir.mkdir(exist_ok=True, parents=True)
    charsiu_textgrid_dir.mkdir(exist_ok=True, parents=True)

    iterator = tqdm.tqdm(
        textgrid_files,
        desc="Copying charsiu textgrid files to datasets directory",
        total=len(textgrid_files),
        dynamic_ncols=True
    )

    for textgrid_file in iterator: #TODO this is unbearably slow, can we make it faster please?
        if textgrid_file.stem in found_stems:
            cp(textgrid_file, charsiu_textgrid_dir / (textgrid_file.stem + '.textgrid'))
    
    iterator = tqdm.tqdm(
        mp3_found,
        desc="Converting charsiu mp3 files to wav, and writing to datasets directory",
        total=len(mp3_found),
        dynamic_ncols=True
    )

    for mp3_file in iterator:
        audio = ppgs.load.audio(mp3_file)
        torchaudio.save(charsiu_wav_dir / (mp3_file.stem + '.wav'), audio, sample_rate=ppgs.SAMPLE_RATE)
