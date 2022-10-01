from multiprocessing.sharedctypes import Value
from nltk import download, corpus, data
from nltk.tokenize import TweetTokenizer
from ppgs.preprocess.words.align import align_one_to_many
import pypar
import csv
import tqdm
from pathlib import Path

try:
    data.find('tokenizers/punkt')
except LookupError:
    download('punkt')

try:
    lookup = corpus.cmudict.dict()
except LookupError:
    download('cmudict')
    lookup = corpus.cmudict.dict()

tokenizer = TweetTokenizer()

def remove_non_alpha(string):
    return ''.join([char for char in string if char.isalpha()])



def get_word_phones(word):
    #TODO handle hyphens and unusual compound words
    #TODO figure out what to do for unusual names
    try:
        pronunciations = lookup[word.lower()]
    except KeyError:
        if '-' in word:
            word_parts = word.split('-')
            return get_word_phones(word_parts[0]) + get_word_phones(word_parts[1])
        elif word[-2:] == "'s":
            return get_word_phones(word[:-2]) + ['S']
        else:
            raise KeyError(word)

    pronunciations_cleaned = [[remove_non_alpha(phn).lower() for phn in pro] for pro in pronunciations]

    return pronunciations_cleaned



def word_align_phones(word_seq, phone_seq):
    """Uses CMUDICT to align phoneme sequences to word sequences"""
    
    word_seq_phones = [get_word_phones(word) for word in word_seq]

    word_seq_phones = [phones[0] for phones in word_seq_phones]
    #TODO explore using multiple phonetic transcriptions per word

    # print(word_seq_phones)
    # print(phone_seq)

    alignment = align_one_to_many(
        word_seq,
        {word_seq[i]: word_seq_phones[i] for i in range(len(word_seq))},
        phone_seq,
        as_splits=True
    )

    return alignment



def from_sequence_data(phone_seq, phone_start, phone_stop, word_seq=None):
    if word_seq:

        last_stop = phone_stop[-1]

        idx = 0
        while idx < len(phone_seq):
            if phone_seq[idx] == 'pau':
                del phone_seq[idx]
                del phone_start[idx]
                del phone_stop[idx]
            else:
                idx += 1

        alignment = word_align_phones(word_seq, phone_seq)
        assert len(alignment) == len(word_seq) + 1
        word_objects = []
        for i in range(1, len(alignment)):
            phone_objects = []
            for j in range(alignment[i-1], alignment[i]):
                phone_object = pypar.Phoneme(phone_seq[j], phone_start[j], phone_stop[j])
                phone_objects.append(phone_object)
            word_object = pypar.Word(word_seq[i-1], phone_objects)
            word_object.validate()
            word_objects.append(word_object)

        #add silences back in
        silences = []
        for i in range(0, len(word_objects)+1):
            if i==0: #for preceding silence
                prior = 0
            else:
                prior = word_objects[i-1].end()
            if i == len(word_objects): #for trailing silence
                current = last_stop
            else:
                current = word_objects[i].start()
            if current - prior > 1e-3:
                silences.append((pypar.Word(pypar.SILENCE, [pypar.Phoneme(pypar.SILENCE, prior, current)]), i))
        for silence, idx in reversed(silences):
            word_objects.insert(idx, silence)

        # for word_obj in word_objects:
            # print(word_obj.word, word_obj.start(), word_obj.end())

        alignment_object = pypar.Alignment(word_objects)
        # for word_obj in word_objects:
            # print(word_obj.word, word_obj.start(), word_obj.end())
        return alignment_object
        
    else:
        word_objects = []
        for i in range(0, len(phone_seq)):
            word_objects.append(pypar.Word(phone_seq[i], 
                                [pypar.Phoneme(phone_seq[i], phone_start[i], phone_stop[i])]))
        return pypar.Alignment(word_objects)



def from_file(phone_file, prompt=None):
    words = None
    if prompt is not None:
        words = tokenizer.tokenize(prompt)
        words = [word.lower() for word in words if not (len(word) == 1 and not word.isalpha())]

    with open(phone_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) #skip header
        phone_stop, phone_seq = list(map(list, zip(*reader)))
        phone_stop = [float(stop) for stop in phone_stop]
        phone_start = [0] + phone_stop[:-1]
        
    alignment = from_sequence_data(phone_seq, phone_start, phone_stop, word_seq=words)
    return alignment



def from_file_to_file(phone_file, output_file, prompt=None):
    alignment = from_file(phone_file, prompt=prompt)
    # alignment.save(output_file)

def from_files_to_files(phone_files, output_files, prompt_file=None):
    print(phone_files[0], output_files[0], prompt_file)
    prompts = None
    if prompt_file is not None:
        with open(prompt_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) #skip header
            
            prompts = {k: v for k, v in reader}

    iterator = tqdm.tqdm(
        zip(phone_files, output_files),
        desc='Creating phoneme alignment representation',
        total=len(phone_files),
        dynamic_ncols=True
    )

    failed_alignments = []
    for phone_file, output_file in iterator:
        prompt = None
        if prompts:
            try:
                prompt = prompts[phone_file.stem]
            except KeyError:
                failed_alignments.append((phone_file.stem, 'Prompt lookup'))
                continue
        try:
            from_file_to_file(phone_file, output_file, prompt=prompt)
        except KeyError as e:
            failed_alignments.append((phone_file.stem, e))
        except ValueError:
            failed_alignments.append((phone_file.stem, 'Alignment failure'))
    # print(failed_alignments)
    with open('/home/cameron/ppgs/logs.csv', 'w') as f:
        writer = csv.writer(f)
        for failure in failed_alignments:
            writer.writerow(list(failure))

            


if __name__ == '__main__':
    # alignment = from_sequence_data(
    #     [
    #         'AO',
    #         'TH',
    #         'ER',
    #         'AH',
    #         'V',
    #         'DH',
    #         'AX',
    #         'D',
    #         'EY',
    #         'N',
    #         'JH',
    #         'ER',
    #         'T',
    #         'R',
    #         'EY',
    #         'L'
    #     ],
    #     range(0, 16),
    #     range(1, 17),
    #     [
    #         'author',
    #         'of',
    #         'the',
    #         'danger',
    #         'trail'
    #     ],
    # )

    test_path = Path('/home/cameron/ppgs/data/datasets/arctic/cmu_us_jmk_arctic/')
    # print(from_file('/home/cameron/ppgs/data/datasets/arctic/cmu_us_awb_arctic/lab/arctic_a0186.csv', "Don't you see., I'm chewing this thing in two."))
    from_files_to_files(
    [test_path / 'lab/arctic_a0046.csv'], 
    ['/home/cameron/ppgs/arctic_test.json'], 
    test_path / 'sentences.csv')
    