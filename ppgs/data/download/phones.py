import ppgs

def timit_to_arctic_phone(timit_phone):
    try:
        return ppgs.TIMIT_TO_ARCTIC_MAPPING[timit_phone.lower()]
    except KeyError:
        return 'unk' #unknown


def timit_to_arctic_phone_seq(timit_phone_seq):
    return [timit_to_arctic_phone(phone) for phone in timit_phone_seq]

def sample_to_seconds(sample, sample_rate=16000):
    return int(sample) / sample_rate

def samples_to_seconds(samples, sample_rate=16000):
    return [sample_to_seconds(sample, sample_rate) for sample in samples]

def timit_to_arctic(rows, backfill=True):
    transposed = list(zip(*rows))
    phones = transposed[2]
    phone_stops = transposed[1]
    mapped_phones = timit_to_arctic_phone_seq(phones)

    if backfill:
        backfill_indices = [idx for idx, phone in enumerate(mapped_phones) if phone == 'bck']
        phones = [phone for phone in mapped_phones if phone != 'bck']
        phone_stops = [stop for idx, stop in enumerate(phone_stops) if idx not in backfill_indices]

        #convert sample counts to seconds
        phone_stops = samples_to_seconds(phone_stops)

    return list(zip(phone_stops, phones))

