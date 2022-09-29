from nltk import download, corpus
from ppgs.preprocess.words.align import align_one_to_many

try:
    lookup = corpus.cmudict.dict()
except LookupError:
    download('cmudict')
    lookup = corpus.cmudict.dict()

def remove_non_alpha(string):
    return ''.join([char for char in string if char.isalpha()])

def get_word_phones(word):
    pronunciations = lookup[word.lower()]

    pronunciations_cleaned = [[remove_non_alpha(phn) for phn in pro] for pro in pronunciations]

    return pronunciations_cleaned


def word_align_phones(word_seq, phone_seq):
    """Uses CMUDICT to align phoneme sequences to word sequences"""
    
    word_seq_phones = [get_word_phones(word) for word in word_seq]

    word_seq_phones = [phones[0] for phones in word_seq_phones]
    #TODO explore using multiple phonetic transcriptions per word

    alignment = align_one_to_many(
        word_seq,
        {word_seq[i]: word_seq_phones[i] for i in range(len(word_seq))},
        phone_seq
    )

    return alignment


if __name__ == '__main__':
    alignment = word_align_phones([
        'author',
        'of',
        'the',
        'danger',
        'trail'
    ],
    [
        'AO',
        'TH',
        'ER',
        'AH',
        'V',
        'DH',
        'AX',
        'D',
        'EY',
        'N',
        'JH',
        'ER',
        'T',
        'R',
        'EY',
        'L'
    ])

    print(alignment)

    