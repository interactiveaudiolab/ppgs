import torch


###############################################################################
# Dataloader
###############################################################################


class Collate():

    def __init__(self, features=['audio']):
        self.features = features

    def __call__(self, batch):
        batch_values = []
        batch_size = len(batch)
        for feature, values in zip(self.features, zip(*batch)):

            # Pack audio
            if feature == 'audio':
                max_length = max([audio.shape[-1] for audio in values])
                padded_audio = torch.zeros(
                (batch_size, 1, max_length),
                dtype=torch.float)
                for i, audio in enumerate(values):
                    padded_audio[i, 0, :audio.shape[-1]] = audio[0]
                batch_values.append(padded_audio)

            # Pack target phonemes
            elif feature == 'phonemes':
                max_length = max([indices.shape[-1] for indices in values])
                padded_indices = torch.full(
                    (batch_size, max_length),
                    -100,
                    dtype=torch.long)
                for i, indices in enumerate(values):
                    padded_indices[i, :indices.shape[-1]] = indices
                batch_values.append(padded_indices)

            # Pack stem
            elif feature == 'stem':
                batch_values.append(values)

            # Pack filename
            elif feature == 'audio_file':
                batch_values.append(values)

            # Pack lengths
            elif feature == 'length':
                batch_values.append(torch.tensor(values))

            # Pack input audio representation
            else:
                max_length = max([latent.shape[-1] for latent in values])
                padded_latents = torch.zeros(
                    (batch_size,) + values[0].shape[:-1] + (max_length,),
                    dtype=values[0].dtype)
                for i, latent in enumerate(values):
                    padded_latents[i, ..., :latent.shape[-1]] = latent
                batch_values.append(padded_latents)

        return batch_values
