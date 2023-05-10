from ppgs.load import audio
from ppgs import CACHE_DIR, DATA_DIR
from tqdm import tqdm
import sys

args = sys.argv

num_threads = 4

dir = CACHE_DIR / args[1]

print('globbing for audio files')

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
    a = audio(audio_file)
    total_num_samples += a.shape[-1]
    total_duration_seconds += a.shape[-1] / 16000

print('total number of samples:', total_num_samples)
print('total duration in seconds:', total_duration_seconds)
print('total duration in minutes:', total_duration_seconds / 60)
print('total duration in hours:', total_duration_seconds / 60 / 60)
print('total duration in days:', total_duration_seconds / 60 / 60 / 24)
