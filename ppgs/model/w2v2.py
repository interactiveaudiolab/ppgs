import torch
import transformers

import ppgs

# Turn off logging
transformers.utils.logging.set_verbosity_error()


###############################################################################
# Pretrained wav2vec 2.0 model
###############################################################################


class W2V2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.padding = 400 // 2 - 160 // 2

        # Load model
        self.w2v2 = transformers.Wav2Vec2Model.from_pretrained(
            'facebook/wav2vec2-base')

        # Upsample by editing the stride
        self.w2v2.feature_extractor.conv_layers[-1].conv.stride = (1,)

        # Freeze feature extractor
        self.w2v2.freeze_feature_extractor()

        # Project onto space of phonemes
        assert ppgs.KERNEL_SIZE % 2 == 1
        self.output_projection = torch.nn.Conv1d(
            in_channels=768,
            out_channels=ppgs.OUTPUT_CHANNELS,
            kernel_size=ppgs.KERNEL_SIZE,
            padding=ppgs.KERNEL_SIZE // 2)

    def forward(self, input_tensor, lengths):
        # Pad input
        padded = torch.nn.functional.pad(
            input_tensor,
            (self.padding, self.padding)
        ).squeeze(dim=1)

        # Create mask
        mask = ppgs.model.transformer.mask_from_lengths(
            lengths,
            self.padding
        ).squeeze(dim=1).to(torch.long)

        # Infer
        w2v2_latent = self.w2v2(padded, mask).last_hidden_state
        w2v2_latent = torch.transpose(w2v2_latent, 1, 2)
        return self.output_projection(w2v2_latent)
