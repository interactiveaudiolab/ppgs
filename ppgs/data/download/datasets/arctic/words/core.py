from nltk import download, corpus, data
from nltk.tokenize import TweetTokenizer
from .align import align_one_to_many
# import pypar
import csv
import tqdm

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
    #TODO handle unusual compound words
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

    alignment = align_one_to_many(
        word_seq,
        {word_seq[i]: word_seq_phones[i] for i in range(len(word_seq))},
        phone_seq,
        as_splits=True
    )

    return alignment

def from_sequence_data(phone_seq, phone_start, phone_end, word_seq=None):
    if word_seq:

        last_stop = phone_end[-1]

        idx = 0
        while idx < len(phone_seq):
            if phone_seq[idx] == 'pau':
                del phone_seq[idx]
                del phone_start[idx]
                del phone_end[idx]
            else:
                idx += 1

        alignment = word_align_phones(word_seq, phone_seq)
        assert len(alignment) == len(word_seq) + 1
        words = []
        for i in range(1, len(alignment)):
            word_start = phone_start[alignment[i-1]]
            word_end = phone_end[alignment[i]-1]
            words.append([word_start, word_end, word_seq[i-1]])

        #add silences back in
        silences = []
        for i in range(0, len(words)+1):
            if i==0: #for preceding silence
                prior = 0
            else:
                prior = words[i-1][1] #previous end
            if i == len(words): #for trailing silence
                current = last_stop
            else:
                current = words[i][0] #current start
            if current - prior > 1e-3:
                silences.append(([prior, current, 'pau'], i))
        for silence, idx in reversed(silences):
            words.insert(idx, silence)

        return words
        
    else:
        return zip(phone_start, phone_end, phone_seq)



def from_file(phone_file, prompt=None):
    words = None
    if prompt is not None:
        words = tokenizer.tokenize(prompt)
        words = [word.lower() for word in words if not (len(word) == 1 and not word.isalpha())]

    with open(phone_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) #skip header
        phone_end, phone_seq = list(map(list, zip(*reader)))
        phone_end = [float(stop) for stop in phone_end]
        phone_start = [0] + phone_end[:-1]
        
    alignment = from_sequence_data(phone_seq, phone_start, phone_end, word_seq=words)
    return alignment

def from_file_to_file(phone_file, output_file, prompt=None):
    alignment = from_file(phone_file, prompt=prompt)
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'word'])
        writer.writerows(alignment)

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
                failed_alignments.append((phone_file.stem, phone_file.parents[1].stem, 'Prompt lookup'))
                from_file_to_file(phone_file, output_file)
                continue
        try:
            from_file_to_file(phone_file, output_file, prompt=prompt)
        except KeyError as e:
            failed_alignments.append((phone_file.stem, phone_file.parents[1].stem, e))
            from_file_to_file(phone_file, output_file)
        except ValueError:
            failed_alignments.append((phone_file.stem, phone_file.parents[1].stem, 'Alignment failure'))
            from_file_to_file(phone_file, output_file)
    # print(failed_alignments)
    