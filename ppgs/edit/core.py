import struct
import re

def edit_phone(ppg, old_phone, new_phone):
    """edit a single phoneme by one-way reallocation of probability"""
    old_index = ppgs.PHONEME_LIST.index(old_phone)
    new_index = ppgs.PHONEME_LIST.index(new_phone)
    ppg[new_index, :] += ppg[old_index, :]
    ppg[old_index, :] = 0

def subseq_search(seq, subseq):
    """A bit of regex and encoding abuse to match subsequences of phonemes in a sequence"""
    pattern = re.escape(struct.pack('b'*len(subseq), *subseq))
    string = struct.pack('b'*len(seq), *seq)
    matches = re.finditer(pattern, string)
    return [m.span()[0] for m in matches]

def edit_phone_seq(ppg, old_phone_seq, new_phone_seq):
    """Replace all occurences of a sequence of phonemes with another sequence of phonemes by swapping probabilities"""
    old_seq_indices = [ppgs.PHONEME_LIST.index(phone) for phone in old_phone_seq]
    new_seq_indices = [ppgs.PHONEME_LIST.index(phone) for phone in new_phone_seq]
    assert len(old_seq_indices) == len(new_seq_indices)
    indices = ppg.argmax(dim=0)
    unique_indices, inverse = torch.unique_consecutive(indices, return_inverse=True)
    match_indices = torch.tensor(subseq_search(unique_indices, old_seq_indices))

    for i in range(0, len(old_phone_seq)):
        slicing = torch.isin(inverse, match_indices+i)
        temporary = ppg[new_seq_indices[i], slicing].clone()
        ppg[new_seq_indices[i], slicing] = ppg[old_seq_indices[i], slicing]
        ppg[old_seq_indices[i], slicing] = temporary

def reinforce_phone(ppg, phone):
    """Allocate all probability to the given phone when it is maximal"""
    index = ppgs.PHONEME_LIST.index(phone)

    indices = ppg.argmax(dim=0)

    ppg[:, indices == index] = 0.0
    ppg[index, indices == index] = 1.0

def get_phone_seq(ppg):
    """Return the sequence of phonemes present in a ppg"""
    indices = ppg.argmax(dim=0)
    unique_indices, inverse = torch.unique_consecutive(indices, return_inverse=True)

    return [ppgs.PHONEME_LIST[i] for i in unique_indices]