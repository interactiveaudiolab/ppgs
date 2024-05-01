import csv

import torchutil

import ppgs


###############################################################################
# Arctic text data wrangling
###############################################################################


def from_sequence_data(phone_seq, phone_start, phone_end, word_seq=None):
    """Align words and phonemes, and fill gaps with silence"""
    if word_seq:

        # Get audio duration
        duration = phone_end[-1]

        # Remove silence
        idx = 0
        while idx < len(phone_seq):
            if phone_seq[idx] == 'pau':
                del phone_seq[idx]
                del phone_start[idx]
                del phone_end[idx]
            else:
                idx += 1

        # Align phoneme and word sequences
        alignment = word_align_phones(word_seq, phone_seq)
        assert len(alignment) == len(word_seq) + 1

        # Extract word alignment
        words = []
        for i in range(1, len(alignment)):
            word_start = phone_start[alignment[i-1]]
            word_end = phone_end[alignment[i]-1]
            words.append([word_start, word_end, word_seq[i-1]])

        # Add silences back in
        silences = []
        for i in range(0, len(words)+1):

            # Get start time
            if i == 0:
                prior = 0
            else:
                prior = words[i-1][1]

            # Get end time
            if i == len(words):
                current = duration
            else:
                current = words[i][0]

            # Fill gaps with silence
            if current - prior > 1e-3:
                silences.append(([prior, current, 'pau'], i))

        # Update alignment
        for silence, idx in reversed(silences):
            words.insert(idx, silence)

        return words

    else:

        return zip(phone_start, phone_end, phone_seq)


def from_file(phone_file, prompt=None):
    """Align words and phonemes from files, and fill gaps with silence"""
    # Maybe tokenize text to retrieve words
    if prompt is not None:

        # Maybe cache tokenizer
        if not hasattr(from_file, 'tokenizer'):
            import nltk

            # Download tokenizer if necessary
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            # Cache tokenizer
            from_file.tokenizer = nltk.tokenize.TweetTokenizer()

        # Tokenize
        words = from_file.tokenizer.tokenize(prompt)

        # Lint
        words = [
            word.lower() for word in words
            if not (len(word) == 1 and not word.isalpha())]

    else:
        words = None

    # Load phoneme alignment
    with open(phone_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) #skip header
        phone_end, phone_seq = list(map(list, zip(*reader)))
        phone_end = [float(stop) for stop in phone_end]
        phone_start = [0] + phone_end[:-1]

    # Align phonemes and words
    return from_sequence_data(phone_seq, phone_start, phone_end, words)


def from_file_to_file(phone_file, output_file, prompt=None):
    """Align words and phonemes from files, fill gaps with silence, and save"""
    # Align
    alignment = from_file(phone_file, prompt=prompt)

    # Save
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'word'])
        writer.writerows(alignment)


def from_files_to_files(phone_files, output_files, prompt_file=None):
    """Align words and phonemes from files, fill gaps with silence, and save"""
    prompts = None
    if prompt_file is not None:
        with open(prompt_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) #skip header
            prompts = {k: v for k, v in reader}

    for phone_file, output_file in torchutil.iterator(
        zip(phone_files, output_files),
        'Creating phoneme alignment representation',
        total=len(phone_files)
    ):
        if prompts:

            try:

                # Get transcript
                prompt = prompts[phone_file.stem]

                # Align
                from_file_to_file(phone_file, output_file, prompt=prompt)

            except (KeyError, ValueError):

                # Recover words from phonemes and align
                from_file_to_file(phone_file, output_file)


###############################################################################
# Utilities
###############################################################################


def get_word_phones(word):
    """Convert word to phonemes using CMU dictionary"""
    if not hasattr(get_word_phones, 'lookup'):
        import nltk
        try:
            get_word_phones.lookup = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            get_word_phones.lookup = nltk.corpus.cmudict.dict()


    try:

        # Query dictionary for word
        pronunciations = get_word_phones.lookup[word.lower()]

    except KeyError:

        # Try again after some common linting
        if '-' in word:
            word_parts = word.split('-')
            return (
                get_word_phones(word_parts[0]) +
                get_word_phones(word_parts[1]))
        elif word[-2:] == "'s":
            return get_word_phones(word[:-2]) + ['S']

        # Give up
        else:
            raise KeyError(word)

    # Remove numerics from phoneme labels
    return [
        [''.join([c for c in phn if c.isalpha()]).lower() for phn in pro]
        for pro in pronunciations]


def word_align_phones(word_seq, phone_seq):
    """Align phoneme sequences to word sequences"""
    # Get corresponding phonemes
    word_seq_phones = [get_word_phones(word)[0] for word in word_seq]

    # Align phonemes and words
    return ppgs.data.datasets.arctic.words.align.align_one_to_many(
        word_seq,
        {word_seq[i]: word_seq_phones[i] for i in range(len(word_seq))},
        phone_seq,
        as_splits=True)
