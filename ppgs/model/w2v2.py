from transformers import Wav2Vec2Model
from transformers.utils import logging
import torch
from ppgs.model.transformer import mask_from_lengths
import ppgs

logging.set_verbosity_error()

class W2V2(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Load model
        self.w2v2: Wav2Vec2Model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

        # Charsiu trick to upsample to 10ms
        self.w2v2.feature_extractor.conv_layers[-1].conv.stride = (1,)

        self.w2v2.freeze_feature_extractor()

        # Project onto space of phonemes
        assert ppgs.KERNEL_SIZE % 2 == 1
        self.output_projection = torch.nn.Conv1d(
            in_channels=768,
            out_channels=len(ppgs.PHONEME_LIST),
            kernel_size=ppgs.KERNEL_SIZE,
            padding=ppgs.KERNEL_SIZE // 2
        )

        self.offset = ppgs.preprocess.w2v2ft.WINDOW_SIZE // 2 - ppgs.preprocess.w2v2ft.HOP_SIZE // 2

    def forward(self, input_tensor: torch.Tensor, lengths: torch.Tensor):
        mask = mask_from_lengths(lengths, self.offset).squeeze(dim=1).to(torch.long)
        w2v2_latent = self.w2v2(input_tensor, mask).last_hidden_state
        w2v2_latent = torch.transpose(w2v2_latent, 1, 2)
        ppg = self.output_projection(w2v2_latent)
        return ppg