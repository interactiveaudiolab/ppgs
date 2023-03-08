from pathlib import Path
from shutil import move
from tqdm import tqdm

files = list(Path('data/cache/charsiu/').glob('*-w2v2.pt'))

iterator = tqdm(
    files,
    "moving files",
    len(files),
    dynamic_ncols=True
)

for file in iterator:
    move(file, file.parent / (file.name[:-7] + 'w2v2fs.pt'))