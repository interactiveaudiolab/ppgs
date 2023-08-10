import torch
import torchaudio
from ppgs import CACHE_DIR, DATA_DIR
# from promonet import CACHE_DIR, DATA_DIR
from tqdm import tqdm
import sys

args = sys.argv

num_threads = 4

dir = CACHE_DIR / args[1]

print(f'globbing for audio files in {dir}')

audio_files = list(dir.rglob('*.wav'))

print('number of audio_files:', len(audio_files))

iterator = tqdm(
    audio_files,
    desc='scanning audio files',
    total=len(audio_files),
    dynamic_ncols=True
)


total_num_samples = 0
total_duration_seconds = 0
for audio_file in iterator:
    num_frames = torchaudio.info(audio_file).num_frames
    total_num_samples += num_frames
    total_duration_seconds += num_frames / 16000

print('total number of samples:', total_num_samples)
print('total duration in seconds:', total_duration_seconds)
print('total duration in minutes:', total_duration_seconds / 60)
print('total duration in hours:', total_duration_seconds / 60 / 60)
print('total duration in days:', total_duration_seconds / 60 / 60 / 24)
