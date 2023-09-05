import ppgs
import pypar
from tqdm import tqdm
from shutil import copy
from pathlib import Path

cache_dir = ppgs.CACHE_DIR
alignment_files = list(cache_dir.rglob('*.[Tt]ext[Gg]rid'))
print(f"found {len(alignment_files)} alignments")


target_word = 'tomato'.lower()

alignment_files_containing_tomato = []
# found_words = set()
iterator = tqdm(alignment_files, total=len(alignment_files), desc='scanning for tomatoes')
for alignment_file in iterator:
# for alignment_file in alignment_files:
    alignment = pypar.Alignment(alignment_file)
    words = [str(word).lower() for word in alignment.words()]
    # for word in words: found_words.add(word)
    if target_word in words:
        alignment_files_containing_tomato.append(alignment_file)
print(f'found {len(alignment_files_containing_tomato)} alignment files containing the word "{target_word}"')
print(alignment_files_containing_tomato)

output_dir = Path('tmp/') / 'words' / target_word
output_dir.mkdir(exist_ok=True, parents=True)

for alignment_file in alignment_files_containing_tomato:
    copy(alignment_file, output_dir / alignment_file.name)
    copy(alignment_file.parent / (alignment_file.stem + '.wav'), output_dir / (alignment_file.stem + '.wav'))