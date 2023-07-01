import re

def v0_90_to_v0_95(id, v90_sentences, v95_sentences):
    """maps cmu_arctic data ids from version 0.90 to version 0.95 by taking an id and the
    contents of a prompt file for v90 and a prompt_file for v95"""
    sentence = re.search(rf'\( {id} \"(.+)\" \)', v90_sentences).groups()[0]
    try:
        new_id = re.search(rf'\( (arctic_[ab][0-9][0-9][0-9][0-9]) \"{sentence}\" \)', v95_sentences).groups()[0]
    except AttributeError:
        return None
    return new_id