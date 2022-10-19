import csv
import pypar
import tqdm
import numpy as np

def from_sequence_data(phone_timings_seq, word_timings_seq):
    #convert pau to sp (or is it vice versa?)
    phone_timings_seq = [[start, end, phone] if phone != 'pau' else [start, end, pypar.SILENCE] for start, end, phone in phone_timings_seq]
    word_timings_seq = [[start, end, word] if word != 'pau' else [start, end, pypar.SILENCE] for start, end, word in word_timings_seq]

    #assure no gaps
    #no gap at front
    if word_timings_seq[0][0] > 0:
        word_timings_seq.insert(0, [0, word_timings_seq[0][0], pypar.SILENCE])
    i = 0
    #no gaps in between
    while i < len(word_timings_seq):
        if word_timings_seq[i-1][1] < word_timings_seq[i][0]:
            word_timings_seq.insert(i, [word_timings_seq[i-1][1], word_timings_seq[i][0], pypar.SILENCE])
            i += 1
        i += 1
    #no gaps at rear
    if word_timings_seq[-1][1] < phone_timings_seq[-1][1]: #here we assume last phone is silence
        try:
            assert phone_timings_seq[-1][2] == pypar.SILENCE
        except AssertionError:
            raise ValueError(phone_timings_seq, word_timings_seq)
        word_timings_seq.append([word_timings_seq[-1][1], phone_timings_seq[-1][1], pypar.SILENCE])


    #put phones into word buckets
    phone_bucket_indices = [-1] * len(phone_timings_seq)
    for i, phone in enumerate(phone_timings_seq):
        for j, word in enumerate(word_timings_seq):
            if phone[0] >= word[0] and phone[1] <= word[1]:
                phone_bucket_indices[i] = j

    if -1 in phone_bucket_indices:
        unaligned_phone_idx = phone_bucket_indices.index(-1)
        unaligned_phone = phone_timings_seq[unaligned_phone_idx]
        if unaligned_phone[2] == pypar.SILENCE: #handle pause mis-alignment
            if unaligned_phone_idx == 0:
                second_phone = phone_timings_seq[1]
                first_word = word_timings_seq[0]
                phone_timings_seq[0] = [0, first_word[1], pypar.SILENCE]
                phone_timings_seq.insert(1, [first_word[1], second_phone[0], pypar.SILENCE])
            elif unaligned_phone_idx == len(phone_bucket_indices) - 1:
                second_to_last_phone = phone_timings_seq[-2]
                last_word = word_timings_seq[-1]
                phone_timings_seq[-1] = [last_word[0], last_word[1], pypar.SILENCE]
                phone_timings_seq.insert(-1, [second_to_last_phone[0], last_word[0], pypar.SILENCE])
            else:
                prev_word_idx = phone_bucket_indices[unaligned_phone_idx-1]
                next_word_idx = prev_word_idx + 1
                prev_word = word_timings_seq[prev_word_idx]
                next_word = word_timings_seq[next_word_idx]
                phone_timings_seq[unaligned_phone_idx] = [unaligned_phone[0], prev_word[1], pypar.SILENCE]
                phone_timings_seq.insert(unaligned_phone_idx+1, [next_word[0], unaligned_phone[1], pypar.SILENCE])
                return from_sequence_data(phone_timings_seq, word_timings_seq) #recurse with split silence


    #Assure all phones in a bucket
    if -1 in phone_bucket_indices:
        unaligned_phone_idx = phone_bucket_indices.index(-1)
        unaligned_phone = phone_timings_seq[unaligned_phone_idx]
        raise ValueError(f'Unaligned phone {unaligned_phone[2]} at index {unaligned_phone_idx} \n with {word_timings_seq} \n\n {phone_timings_seq}')

    #Assure monotonic phone mapping
    for i in range(1, len(phone_bucket_indices)):
        assert phone_bucket_indices[i-1] <= phone_bucket_indices[i]

    #create pypar alignment object starting with phones, then words, then alignment
    phone_objects = [pypar.Phoneme(phone, start, end) for start, end, phone in phone_timings_seq]

    word_objects = []
    for i in range(0, len(word_timings_seq)):
        try:
            start_phone_idx = phone_bucket_indices.index(i)
        except ValueError:
            continue
            raise ValueError ("word has no corresponding phones", phone_timings_seq, word_timings_seq, i)
        end_phone_idx = -1 * list(reversed(phone_bucket_indices)).index(i) - 1

        phone_slice = slice(start_phone_idx, end_phone_idx+1 if end_phone_idx < -1 else None)
        word_objects.append(pypar.Word(word_timings_seq[i][2], phone_objects[phone_slice]))

    alignment_obj = pypar.Alignment(word_objects)
    
    #Check for missing phones
    for timestep in np.arange(0, alignment_obj.duration(), 0.001):
        if alignment_obj.phoneme_at_time is None:
            raise ValueError(f'no phone at time {timestep}, {alignment_obj.words()}, {alignment_obj.phonemes()}')
    return alignment_obj



def from_file(phone_file, word_file):
    with open(phone_file) as f:
        reader = csv.reader(f)
        next(reader) #skip header
        phone_timings_seq = list(reader)
    with open(word_file) as f:
        reader = csv.reader(f)
        next(reader) #skip header
        word_timings_seq = list(reader)
    phone_timings_seq = [(float(prev_phone[0]), float(curr_phone[0]), curr_phone[1]) for prev_phone, curr_phone in zip([[0, 0]] + phone_timings_seq, phone_timings_seq + [[0, 0]])][:-1]
    word_timings_seq = [(float(start), float(end), word) for start, end, word in word_timings_seq]
    return from_sequence_data(phone_timings_seq, word_timings_seq)



def from_file_to_file(phone_file, word_file, output_file):
    from_file(phone_file, word_file).save(output_file)



def from_files_to_files(phone_files, word_dir, output_dir):
    """converts phone alignment and word alignment files to pypar alignment representation files"""

    #get word and output files
    word_files = []
    output_files = []
    for phone_file in phone_files:
        word_file = word_dir / phone_file.name
        output_file = output_dir / (phone_file.stem + '.textgrid')

        if not word_file.exists():
            raise FileNotFoundError(f'could not find word file {word_file}')
    
        word_files.append(word_file)
        output_files.append(output_file)

    iterator = tqdm.tqdm(
        zip(phone_files, word_files, output_files),
        desc='Creating pypar alignment files',
        total = len(phone_files),
        dynamic_ncols=True
    )

    for phone_file, word_file, output_file in iterator:
        from_file_to_file(phone_file, word_file, output_file)

