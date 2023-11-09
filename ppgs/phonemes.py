import pypar


###############################################################################
# PPG phoneme set
###############################################################################


# Our 40 phoneme categories (in order)
PHONEMES = [
	'aa',
	'ae',
	'ah',
	'ao',
	'aw',
	'ay',
	'b',
	'ch',
	'd',
	'dh',
	'eh',
	'er',
	'ey',
	'f',
	'g',
	'hh',
	'ih',
	'iy',
	'jh',
	'k',
	'l',
	'm',
	'n',
	'ng',
	'ow',
	'oy',
	'p',
	'r',
	's',
	'sh',
	't',
	'th',
	'uh',
	'uw',
	'v',
	'w',
	'y',
	'z',
	'zh',
	pypar.SILENCE]


# Mapping between phonemes and integer category indices
PHONEME_TO_INDEX_MAPPING = {phone: i for i, phone in enumerate(PHONEMES)}


###############################################################################
# Phoneme categorizations
###############################################################################


VOICED = [
	'aa',
	'ae',
	'ah',
	'ao',
	'aw',
	'ay',
	'eh',
	'er',
	'ey',
	'hh',
	'ih',
	'iy',
	'jh',
	'l',
	'm',
	'n',
	'ng',
	'ow',
	'oy',
	'r',
	'uh',
	'uw',
	'v',
	'w',
	'y',
	'z',
	'zh'
]


###############################################################################
# Mappings between phoneme sets
###############################################################################


# The permutation of our phonemes used by Charsiu
CHARSIU_PHONE_ORDER = [
    pypar.SILENCE,
    'ng',
    'f',
    'm',
    'ae',
    'r',
    'uw',
    'n',
    'iy',
    'aw',
    'v',
    'uh',
    'ow',
    'aa',
    'er',
    'hh',
    'z',
    'k',
    'ch',
    'w',
    'ey',
    'zh',
    't',
    'eh',
    'y',
    'ah',
    'b',
    'p',
    'th',
    'dh',
    'ao',
    'g',
    'l',
    'jh',
    'oy',
    'sh',
    'd',
    'ay',
    's',
    'ih']
CHARSIU_PERMUTE = [CHARSIU_PHONE_ORDER.index(phone) for phone in PHONEMES]


# Mapping from the TIMIT phoneme set to our proposed phoneme set
TIMIT_TO_ARCTIC_MAPPING = {
    'aa': 'aa',
    'ae': 'ae',
    'ah': 'ah',
    'ao': 'ao', #differs from Kaldi, likely an error in Kaldi
    'aw': 'aw',
    'ax': 'ah',
    'ax-h': 'ah',
    'axr': 'er',
    'ay': 'ay',
    'b': 'b',
    'bcl': 'bck<b>', #backfill
    'ch': 'ch',
    'd': 'd',
    'dcl': 'bck<d,jh>', #backfill
    'dh': 'dh',
    'dx': 'd', #assumption
    'eh': 'eh',
    'el': 'l',
    'em': 'm',
    'en': 'n',
    'eng': 'ng',
    'epi': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'er': 'er',
    'ey': 'ey',
    'f': 'f',
    'g': 'g',
    'gcl': 'bck<g>', #backfill
    'h#': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'hh': 'hh',
    'hv': 'hh',
    'ih': 'ih',
    'ix': 'ih',
    'iy': 'iy',
    'jh': 'jh',
    'k': 'k',
    'kcl': 'bck<k>', #backfill
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ng',
    'nx': 'n',
    'ow': 'ow',
    'oy': 'oy',
    'p': 'p',
    'pau': pypar.SILENCE, #differs from Kaldi (pau instead of sil)
    'pcl': 'bck<p>', #backfill
    'q': 't', #map to its allophone
    'r': 'r',
    's': 's',
    'sh': 'sh',
    't': 't',
    'tcl': 'bck<t,ch>', #backfill
    'th': 'th',
    'uh': 'uh',
    'uw': 'uw',
    'ux': 'uw',
    'v': 'v',
    'w': 'w',
    'y': 'y',
    'z': 'z',
    'zh': 'zh' #differs from Kaldi
}
