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
    phone_ends = list(transposed[1])
    mapped_phones = timit_to_arctic_phone_seq(phones)

    if backfill:
        backfill_indices = [idx for idx, phone in enumerate(mapped_phones) if phone[:3] == 'bck']
        for i, idx in enumerate(backfill_indices):
            assert mapped_phones[idx][3] == '<' and mapped_phones[idx][-1] == '>'
            possible_replacements = mapped_phones[idx][4:-1].split(',')
            if idx < len(mapped_phones) - 1 and mapped_phones[idx+1] in possible_replacements:
                mapped_phones[idx] = 'bck'
            else:
                mapped_phones[idx] = possible_replacements[0]
        #TODO smarter backfill
        for i in range(0, len(mapped_phones)):
            if mapped_phones[i] == 'bck':
                try:
                    mapped_phones[i] = mapped_phones[i+1]
                except IndexError:
                    raise IndexError("Tried to backfill on last phone in sequence")
        # phones = [phone for phone in mapped_phones if phone != 'bck'] #remove backfilled phones
        # phone_ends = [end for idx, end in enumerate(phone_ends) if idx not in backfill_indices] #remove backfilled end times

        #convert sample counts to seconds
        phone_ends = samples_to_seconds(phone_ends)

    return list(zip(phone_ends, mapped_phones))

