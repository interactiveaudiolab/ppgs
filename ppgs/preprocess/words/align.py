from ctypes import alignment
import numpy as np

#TODO explore
match_score  = 1
mismatch_score = -1
gap_score = -2

#TODO maybe different mismatches can have different scores

def w2p_directional_score(phone0, phone1, direction_idx):
    """Takes two phones and a direction index [up, left, diag] and computes a score.
    Assumes that the first sequence is the one generated from the dict and the second is ground truth."""
    if direction_idx == 0: #moving in just first sequence
        #represents putting a gap in first sequence
        if phone0 == '<end>': #word boundary gap
            return 0
        else: #normal gap
            return -2
    elif direction_idx == 1: #moving in just second sequence
        #represents putting a gap in second sequence
        return -2
    elif direction_idx == 2: #moving in both sequences
        if phone0 == '<end>': #very bad mismatch
            return -4
        elif phone0 == phone1: #match
            return 2
        else: #mismatch
            return -1
    else:
        raise ValueError('expected direction_idx in range(0,3)')


def w2p_score(phone0, phone1):
    """returns the table entry generated from two phonemes"""
    return np.array([w2p_directional_score(phone0, phone1, direction) for direction in range(0, 3)])

def create_table(seq0, seq1, score_function):
    """Create Needleman-Wunsch table and fill in using sequences provided as arguments"""
    #order of third dim is up, left, diag
    #TODO change this to match scoring function
    table = np.fromfunction(lambda x,y,z: -2*(x+y), (len(seq0)+1, len(seq1)+1, 3))


    #fill in table
    for i in range(1, len(seq0)+1):
        for j in range(1, len(seq1)+1):
            phone0 = seq0[i-1]
            phone1 = seq1[j-1]
            table[i,j] = table[[i-1,i,i-1], [j,j-1,j-1]].max(axis=1) + score_function(phone0, phone1)
            # table[i,j,0] = np.max(table[i-1,j]) + gap_score #up
            # table[i,j,1] = np.max(table[i,j-1]) + gap_score #left
            # if seq0[i-1] == seq1[j-1]:
            #     table[i,j,2] = np.max(table[i-1,j-1]) + match_score
            # else:
            #     table[i,j,2] = np.max(table[i-1,j-1]) + mismatch_score
    # print(table.max(axis=2))
    return table

def trace_table(table):
    """"Determine best path back to origin in Needleman-Wunsch table"""
    pos = np.array(table.shape[:2]) - 1
    path = [pos]

    while (pos != np.array([0, 0])).all():
        direction_index = np.argmax(table[pos[0], pos[1]])
        next_pos = [pos-np.array([1,0]), pos-np.array([0,1]), pos-1][direction_index]
        path.append(next_pos)
        pos = path[-1]

    path.append(np.array([0, 0]))

    return list(reversed(path))


def interp_path(path, seq0, seq1):
    """Interpret path through Needleman-Wunsch in terms of the original sequences"""
    out0 = []
    out1 = []
    old = path[0]
    for pos in path[1:]:
        if (pos == old + 1).all():
            out0.append(seq0[pos[0]-1])
            out1.append(seq1[pos[1]-1])
        elif (pos == old + np.array([1, 0])).all():
            out0.append(seq0[pos[0]-1])
            out1.append(None)
        elif (pos == old + np.array([0, 1])).all():
            out0.append(None)
            out1.append(seq1[pos[1]-1])
        old = pos
    return out0, out1


def align_sequences(seq0, seq1, score_function):
    """Align two sequences using Needleman-Wunsch"""
    table = create_table(seq0, seq1, score_function)
    # print(np.max(table, axis=2))
    path = trace_table(table)
    return interp_path(path, seq0, seq1)


def align_one_to_many(one_seq, one_to_many_mapping, many_seq, as_splits=False):
    """
    Align two sequences in a one-to-many context. Align spaces A and B where one symbol in A can map to a sequence of symbols in B.
    Uses Needleman-Wunsch in the B domain.

    Parameters:
        one_seq: sequence containing fewer symbols
        one_to_many_mapping: dictionary mapping one symbol from A to a squence of symbols in B
        many_seq: sequence containing many symbols
        as_splits: if false (default) returns as a list of lists. If true returns as a list of indices to split at.
    """

    # print(one_seq)
    # print(one_to_many_mapping)
    # print(many_seq)

    one_as_many_seq = sum([one_to_many_mapping[symbol] + ['<end>'] for symbol in one_seq], ['<end>'])

    alignment = align_sequences(one_as_many_seq, many_seq, w2p_score)

    split_indices = []
    idx = 0
    while idx < len(alignment[0]):
        if alignment[0][idx] == '<end>':
            if alignment[1][idx] is not None:
                print(alignment[0])
                print(alignment[1])
                raise ValueError('Failed alignment')
            split_indices.append(idx)
            del alignment[0][idx]
            del alignment[1][idx]
        elif alignment[1][idx] is None: #ignore gaps in many_seq
            del alignment[0][idx]
            del alignment[1][idx]
        else:
            idx += 1
    # for idx, token in enumerate(alignment[0]):
    #     if token == '<end>':
    #         if alignment[1][idx] is None:
    #             split_indices.append(idx)
    #         else:
    #             raise ValueError('Failed alignment')

    if as_splits:
        return split_indices

    new_alignment = []
    for i in range(1, len(split_indices)):
        # new_alignment.append(alignment[1][split_indices[i-1]+1:split_indices[i]])
        new_alignment.append(many_seq[split_indices[i-1]:split_indices[i]])

    return new_alignment


if __name__ == '__main__':
    # alignment = align_sequences(
    #     list('ABCDEFBG'),
    #     list('ABCBHEFIG')
    # )
    # print(alignment[0])
    # print(alignment[1])

    alignment = align_one_to_many([
        'author',
        'of',
        'the',
        'danger',
        'trail'
    ],
    {
        'author': ['AO', 'TH', 'ER'],
        'of': ['AH', 'V'],
        'the': ['DH', 'AH'],
        'danger': ['D', 'EY', 'N', 'JH', 'ER'],
        'trail': ['T', 'R', 'EY', 'L']
    },
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