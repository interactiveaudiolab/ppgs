from pathlib import Path
from shutil import move
from tqdm import tqdm
import sys

args = sys.argv

assert len(args) == 3
args = args[1:]
from_name = args[0]
to_name = args[1]

print(f'converting stem {from_name} to {to_name}')

files = list(Path('data/cache/charsiu/').glob(f'*-{from_name}.pt'))

iterator = tqdm(
    files,
    "moving files",
    len(files),
    dynamic_ncols=True
)

for file in iterator:
    dash_position = file.name.rindex('-')
    move(file, file.parent / (file.name[:dash_position] + f'-{to_name}.pt'))