import torch
import ppgs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partition=None):
        audio_dir = ppgs.CACHE_DIR / dataset
        if partition is not None:
            self.audio_files = [audio_dir / (stem + '.wav') for stem in ppgs.load.partition(dataset)[partition]]
        else:
            self.audio_files = list(audio_dir.glob('*.wav'))

    def __getitem__(self, index):
        audio_file = self.audio_files[index]

        audio = ppgs.load.audio(audio_file)

        return audio, audio_file, audio.shape[-1]

    def __len__(self):
        return len(self.audio_files)

def collate(batch):
    audio, audio_files, lengths = zip(*batch)
    max_length = max(lengths)
    batch_size = len(audio_files)
    padded_audio = torch.zeros(
        (batch_size, 1, max_length),
        dtype=torch.float)
    for i, a in enumerate(audio):
        padded_audio[i, 0, :a.shape[-1]] = a[0]
    lengths = torch.tensor(lengths)
    return padded_audio, audio_files, lengths

def loader(dataset, partition=None, batch_size=8, num_workers=0):
    dataset_object = Dataset(dataset, partition)
    loader_object = torch.utils.data.DataLoader(
        dataset=dataset_object,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate
    )
    return loader_object