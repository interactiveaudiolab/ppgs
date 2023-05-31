import csv
from pathlib import Path
from ppgs import CACHE_DIR

root_dir = Path(__file__).parent
tmp_dir = root_dir / 'tmp'
tsv_dir = tmp_dir / 'charsiu_tsv'

tsv_files = list(tsv_dir.glob('*.tsv'))

stem_to_client = {}
for tsv_file in tsv_files:
    with open(tsv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        assert header[0] == 'client_id'
        for row_number, row in enumerate(reader):
            if len(row) == 0:
                continue
            speaker = row[0]
            stem = Path(row[1]).stem
            if stem not in stem_to_client:
                stem_to_client[stem] = speaker
            else:
                assert stem_to_client[stem] == speaker

wav_files = list((CACHE_DIR / 'charsiu').glob('*.wav'))
clients = set()
for wav_file in wav_files:
    stem = wav_file.stem
    try:
        client = stem_to_client[stem]
    except KeyError:
        print(f'failed for file {wav_file}')
    clients.add(client)
print(len(clients))