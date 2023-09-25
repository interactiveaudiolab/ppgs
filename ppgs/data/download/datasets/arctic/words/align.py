import numpy as np


###############################################################################
# Phoneme-word alignment with unknown words
###############################################################################


def align_one_to_many(one_seq, one_to_many_mapping, many_seq, as_splits=False):
    """
    Dynamic programming for one-to-many alignment (Needleman-Wunsch algorithm)

    Arguments:
        one_seq
            Sequence containing fewer symbols
        one_to_many_mapping
            Dictionary mapping one symbol from A to a squence of symbols in B
        many_seq
            Sequence containing many symbols
        as_splits
            If false (default), returns as a list of lists.
            If true, returns as a list of indices to split at.
    """
    # Map sequence tokens
    one_as_many_seq = sum(
        [one_to_many_mapping[symbol] + ['<end>'] for symbol in one_seq],
        ['<end>'])

    # Align sequences
    alignment = needleman_wunsch(
        one_as_many_seq,
        many_seq,
        word_to_phoneme_score)

    # Get split indices
    idx = 0
    split_indices = []
    while idx < len(alignment[0]):
        if alignment[0][idx] == '<end>':
            if alignment[1][idx] is not None:
                raise ValueError('Failed alignment')
            split_indices.append(idx)
            del alignment[0][idx]
            del alignment[1][idx]
        elif alignment[1][idx] is None:
            del alignment[0][idx]
            del alignment[1][idx]
        else:
            idx += 1

    if as_splits:
        return split_indices

    # Split sequence
    return [
        many_seq[split_indices[i - 1]:split_indices[i]]
        for i in range(1, len(split_indices))]


###############################################################################
# Needleman-Wunsch algorithm for sequence alignment
###############################################################################


def needleman_wunsch(seq0, seq1, score_fn):
    """Align two sequences using Needleman-Wunsch"""
    # Forward
    table = forward(seq0, seq1, score_fn)

    # Backward
    path = backward(table)

    # Decode path
    return decode(path, seq0, seq1)


###############################################################################
# Needleman-Wunsch algorithm components
###############################################################################


def backward(table):
    """"Backward pass of the Needleman-Wunsch algorithm"""
    # End point
    pos = np.array(table.shape[:2]) - 1
    path = [pos]

    # Backtrack
    while (pos != np.array([0, 0])).all():

        # Get possible steps
        candidates = [pos - np.array([1, 0]), pos - np.array([0, 1]), pos - 1]

        # Get direction of best step
        index = np.argmax(table[pos[0], pos[1]])

        # Step
        path.append(candidates[index])
        pos = path[-1]

    # Start point
    path.append(np.array([0, 0]))

    # Reverse
    return list(reversed(path))


def decode(path, seq0, seq1):
    """Decode optimal path in terms of the original sequences"""
    out0, out1 = [], []
    old = path[0]
    for pos in path[1:]:
        if (pos == old + 1).all():
            out0.append(seq0[pos[0] - 1])
            out1.append(seq1[pos[1] - 1])
        elif (pos == old + np.array([1, 0])).all():
            out0.append(seq0[pos[0] - 1])
            out1.append(None)
        elif (pos == old + np.array([0, 1])).all():
            out0.append(None)
            out1.append(seq1[pos[1] - 1])
        old = pos
    return out0, out1


def forward(seq0, seq1, score_fn):
    """Forward pass of the Needleman-Wunsch algorithm"""
    # Order of third dim is up, left, diag
    table = np.fromfunction(
        lambda x, y, z: -2 * (x + y),
        (len(seq0) + 1, len(seq1) + 1, 3))

    # Fill in table
    for i in range(1, len(seq0) + 1):
        for j in range(1, len(seq1) + 1):
            phone0 = seq0[i - 1]
            phone1 = seq1[j - 1]
            table[i, j] = (
                table[[i - 1, i, i - 1], [j, j - 1, j - 1]].max(axis=1) +
                score_fn(phone0, phone1))

    return table


def word_to_phoneme_score(phone0, phone1):
    """Computes scores between two phonemes"""
    return np.array([
        word_to_phoneme_directional_score(phone0, phone1, direction)
        for direction in range(0, 3)])


###############################################################################
# Utilities
###############################################################################


def word_to_phoneme_directional_score(phone0, phone1, direction_idx):
    """Computes score between two phonemes given a direction index"""
    # Step forward sequence 0
    if direction_idx == 0:

        # Place a word boundary in sequence 0
        if phone0 == '<end>':
            return 0

        # Place a gap in sequence 0
        else:
            return -2

    # Step forward sequence 1
    elif direction_idx == 1:

        # Place a gap in sequence 1
        return -2

    # Step forward both sequences
    elif direction_idx == 2:

        # Very bad mismatch
        if phone0 == '<end>':
            return -4

        # Match
        elif phone0 == phone1:
            return 2

        # Mismatch
        else:
            return -1

    raise ValueError('expected direction_idx in range(0,3)')
