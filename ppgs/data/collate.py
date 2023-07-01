import torch

def reduced_collate(batch):
    """Batch collation with reduced features"""
    input_ppgs, indices, stems = zip(*batch)
    first_shape = input_ppgs[0].shape
    lengths = torch.tensor([ppg.shape[-1] for ppg in input_ppgs], dtype=torch.long)
    max_length = lengths.max().item()
    index_lengths = torch.tensor([idx.shape[-1] for idx in indices], dtype=torch.long)
    max_index_length = index_lengths.max().item()
    batch = len(input_ppgs)

    padded_ppgs = torch.zeros(
        (batch,) + first_shape[:-1] + (max_length,),
        dtype=torch.float
    )

    # Allocate padded target phoneme indices
    # -100 is the default ignore_index value for cross entropy loss
    padded_indices = torch.full((batch, max_index_length), -100, dtype=torch.long)

    # Populate padded tensors
    for i, (ppg, index) in enumerate(zip(input_ppgs, indices)):
        padded_ppgs[i, ..., :ppg.shape[-1]] = ppg
        padded_indices[i, :index.shape[-1]] = index

    return padded_ppgs, padded_indices, lengths, stems

def collate(batch):
    """Batch collation"""
    # Unpack
    input_ppgs, indices, alignments, word_breaks, waveforms, stems = zip(*batch)

    # Get padded tensor dimensions
    first_shape = input_ppgs[0].shape
    max_length = max([ppg.shape[-1] for ppg in input_ppgs])
    index_lengths = torch.tensor([idx.shape[-1] for idx in indices], dtype=torch.long)
    max_index_length = index_lengths.max().item()
    batch = len(input_ppgs)

    # Allocate padded input ppgs
    padded_ppgs = torch.zeros(
        (batch,) + first_shape[:-1] + (max_length,),
        dtype=torch.float)

    # Allocate padded target phoneme indices
    # -100 is the default ignore_index value for cross entropy loss
    padded_indices = torch.full((batch, max_index_length), -100, dtype=torch.long)

    # Populate padded tensors
    for i, (ppg, index) in enumerate(zip(input_ppgs, indices)):
        padded_ppgs[i, ..., :ppg.shape[-1]] = ppg
        padded_indices[i, :index.shape[-1]] = index

    return padded_ppgs, padded_indices, alignments, word_breaks, waveforms, stems
