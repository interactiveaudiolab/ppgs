from posixpath import normpath
from struct import pack, unpack
import requests
import tarfile
from os import makedirs, walk, listdir
from os.path import exists, join, basename, normpath, isdir, isfile
from shutil import copy as cp
import csv

from ppgs import data

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
        

def datasets(datasets, format_only, timit_source, arctic_speakers):
    """Downloads the datasets passed in"""
    datasets = [dataset.lower() for dataset in datasets]
    if 'timit' in datasets:
        if not format_only:
            download_timit(timit_source)
        else:
            format_timit()
    if 'arctic' in datasets:
        if not format_only:
            download_arctic(arctic_speakers)
        else:
            format_arctic()



def download_timit(timit_source):
    """Prompts user to install TIMIT dataset, and formats dataset if present"""
    makedirs('data/sources', exist_ok=True)
    if timit_source is None:
        possible_paths = [ #in order of preferrence
            'data/sources/timit/',
            'data/sources/timit.tar',
            'data/sources/timit_LDC93S1.tgz',
            'data/sources/timit_LDC93S1.tar.gz',
            'data/sources/timit.tgz',
            'data/sources/timit.tar.gz',
        ]
        source_exists = [exists(path) for path in possible_paths]
        try:
            chosen_source_idx = source_exists.index(True)
        except ValueError:
            raise FileNotFoundError(f"""TIMIT dataset can only be officially downloaded via https://catalog.ldc.upenn.edu/LDC93s1,
            please download this resource and place it in './data/sources/'. This command expects one of {possible_paths} to be present
            or a user provided path using '--timit-source' argument""")
        chosen_source = possible_paths[chosen_source_idx]
    else:
        if not exists(timit_source):
            raise FileNotFoundError(f"User specified file {timit_source} does not exist")
        chosen_source = timit_source
    print(f"Using '{chosen_source}' as source for TIMIT dataset")
    if chosen_source_idx > 0:
        print(f"unzipping {chosen_source} to 'data/sources/'")
        with tarfile.open(chosen_source) as tf:
            #TODO technically unsafe if infected timit source is used (low probability event)
            tf.extractall('data/sources/')
            if not exists(possible_paths[0]):
                raise FileNotFoundError("'data/sources/timit' should exist now, but it does not")
        download_timit()

    format_timit()

def download_arctic(arctic_speakers):
    """Downloads the CMU arctic database"""
    makedirs('data/sources/arctic/', exist_ok=True)
    for arctic_speaker in arctic_speakers:
        if not exists(f"data/sources/arctic/cmu_us_{arctic_speaker}_arctic/"):
            url = f"http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_{arctic_speaker}_arctic-0.95-release.tar.bz2"
            print(f"downloading and unziping arctic data for speaker {arctic_speaker}")
            download_tar_bz2(url, 'data/sources/arctic/')

    format_arctic()

def sph_get_header_size(sphere_file_object):
    sphere_file_object.seek(0)
    assert sphere_file_object.readline() == b'NIST_1A\n'
    header_size = int(sphere_file_object.readline().decode('utf-8')[:-1])
    sphere_file_object.seek(0)
    return header_size

header_type_flag_mapping = {
    'i': int,
    'r': float,
    's': str
}

def sph_get_header(sphere_file_object, header_size):
    sphere_file_object.seek(16)
    header = sphere_file_object.read(header_size-16).decode('utf-8').split('\n')
    header = header[:header.index('end_head')]
    header = [header_item.split(' ') for header_item in header if header_item[0] != ';']
    return {h[0]: header_type_flag_mapping[h[1][1]](h[2]) for h in header}

def wav_make_header(sph_header):
    try:
        return pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            sph_header['sample_count'] * sph_header['sample_n_bytes'] + 36, #file size -8 = datasize + 36
            b'WAVE',
            b'fmt ',
            16, #fmt size
            1, #TODO fill in from header??? #pcm header
            sph_header['channel_count'],  #MONO vs STEREO
            sph_header['sample_rate'],  #sample rate
            sph_header['sample_rate'] * sph_header['sample_n_bytes'], #bytes per second
            sph_header['sample_n_bytes'], #block align (bytes per sample)
            sph_header['sample_n_bytes']*8, #bits per sample
            b'data',
            sph_header['sample_count'] * sph_header['sample_n_bytes'] #data size
        )
    except KeyError as e:
        raise KeyError('key not found in sph header:', e)

def sph_get_samples(sphere_file_object, sphere_header_size):
    sphere_file_object.seek(sphere_header_size)
    return sphere_file_object.read()

#TODO extract to other file
def pcm_sph_to_wav(sphere_file):
    with open(sphere_file, 'rb') as f:
        header_size = sph_get_header_size(f)
        header = sph_get_header(f, header_size)
        new_header = wav_make_header(header)
        samples = sph_get_samples(f, header_size)
        return new_header + samples


def phn_to_lab(phn_file):
    with open(phn_file, 'r') as f:
        return f.read() 


def format_timit():
    """Formats the TIMIT database"""

    #walk filetree and find files
    data_path = 'data/sources/timit/TIMIT/'
    if not exists(data_path):
        raise FileNotFoundError(f"'{data_path}' does not exist")
    sphere_files = []
    text_files = []
    word_files = []
    phone_files = []
    for root, _, files in walk(data_path):
        for file in files:
            file_extension = file[-4:].lower()
            if file_extension == '.wav':
                sphere_files.append((root, file))
            elif file_extension == '.phn':
                phone_files.append((root, file))
            elif file_extension == '.txt':
                text_files.append((root, file))
            elif file_extension == '.word':
                word_files.append((root, file))

    #convert NIST sphere files to WAV format and transfer
    print('Converting sphere files to WAV files for TIMIT dataset')
    for idx, (root, file) in enumerate(sphere_files):
        if idx % (round(len(sphere_files)/10)) == 0:
            print(str(int(idx/(len(sphere_files))*100)) + '%')
        speaker_dir = basename(normpath(root))
        makedirs('data/datasets/timit/' + speaker_dir, exist_ok=True)
        new_path = join('data/datasets/timit/', speaker_dir, file)
        if not exists(new_path):
            with open(new_path, 'wb') as f:
                f.write(pcm_sph_to_wav(join(root, file)))

    #format and transfer sentence information
    print('transferring sentence data for TIMIT dataset')
    raise NotImplementedError('formatting not completely implemented for TIMIT dataset')
    #TODO implement


    #transfer phoneme label files
    print('transferring phoneticl label files for TIMIT dataset')
    raise NotImplementedError('formatting not completely implemented for TIMIT dataset')
    #TODO implement


def format_arctic():
    """Formats the CMU Arctic database"""
    
    data_path = 'data/sources/arctic'
    if not exists(data_path):
        raise FileNotFoundError(f"'{data_path}' does not exist")

    #get arctic speakers
    speakers = [dir for dir in listdir(data_path) if (isdir(join(data_path, dir)) and dir[:3] == 'cmu')]

    #iterate speakers and copy
    for speaker in speakers:
        new_speaker_dir = join('data/datasets/arctic/', speaker)

        #transfer phoneme label files
        lab_dir_path = join(data_path, speaker, 'lab')
        new_lab_dir_path = join(new_speaker_dir, 'lab')

        if not exists(lab_dir_path):
            raise ValueError(f'could not find directory {lab_dir_path}')

        #create destination directory
        makedirs(new_lab_dir_path, exist_ok=True)

        #get label files
        print(f'transferring phonetic label files for arctic speaker {speaker}')
        lab_files = [file for file in listdir(lab_dir_path) if isfile(join(lab_dir_path, file)) and file[-4:] == '.lab']

        for lab_file in lab_files:
            with open(join(lab_dir_path, lab_file), 'r') as f:
                lines = f.readlines()
                non_header_lines = lines[lines.index('#\n')+1:] #get rid of useless headers
                try:
                    timestamps = [line.split()[0] for line in non_header_lines] #extract timestamp info
                    phonemes = [line.split()[2] for line in non_header_lines] #extract phoneme info
                    new_lines = [','.join(z) + '\n' for z in zip(timestamps, phonemes)] #merge info into new lines
                except IndexError:
                    f.seek(0)
                    print(b'{' + f.read().encode('utf-8') + b'}')
                    return
                with open(join(new_lab_dir_path, lab_file.split('.')[0] + '.csv'), 'w') as outf:
                    #write new label file as CSV
                    outf.writelines(['timestamp,phoneme\n'] + new_lines)

        
        #transfer sentence file
        print(f'transferring sentence file for arctic speaker {speaker}')
        sentence_file_path = join(data_path, speaker, 'etc', 'txt.done.data')
        new_sentence_file_path = join(new_speaker_dir, 'sentences.csv')

        if not exists(sentence_file_path):
            raise FileNotFoundError(f'could not find file {sentence_file_path}')

        with open(sentence_file_path, 'r') as f:
            new_file_content = "sample,sentence\n"
            for line in f.readlines():
                assert line[:2] == '( ' and line[-3:] == ' )\n'
                label_and_sentence = line[2:-3].split(maxsplit=1)
                new_file_content += label_and_sentence[0] + ',' + label_and_sentence[1] + '\n'
            with open(new_sentence_file_path, 'w') as outf:
                #write new sentence file as CSV
                outf.write(new_file_content)


        #transfer wav files
        print(f'transferring wav files for arctic speaker {speaker}')
        wav_dir_path = join(data_path, speaker, 'wav')
        new_wav_dir_path = join(new_speaker_dir, 'wav')
        
        if not exists(wav_dir_path):
            raise FileNotFoundError(f'could not find directory {wav_dir_path}')

        makedirs(new_wav_dir_path)

        wav_files = [file for file in listdir(wav_dir_path) if isfile(join(wav_dir_path, file)) and file[-4:] == '.wav']

        for wav_file in wav_files:
            cp(join(wav_dir_path, wav_file), join(new_wav_dir_path, wav_file))


            

        
    