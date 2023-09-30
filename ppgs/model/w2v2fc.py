import torch
import transformers

import ppgs

# Turn off logging
transformers.utils.logging.set_verbosity_error()


###############################################################################
# Charsiu pretrained frame-classification model
###############################################################################


class W2V2FC(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.padding = 400 // 2 - 160 // 2

        # Load pretrained model
        self.w2v2fc = \
            ppgs.preprocess.charsiu_models.Wav2Vec2ForFrameClassification.from_pretrained(
                'charsiu/en_w2v2_fc_10ms')

    def forward(self, input_tensor, lengths):
        # Pad input
        inputs = torch.nn.functional.pad(
            input_tensor,
            (self.padding, self.padding)
        ).squeeze(dim=1)

        # Create mask
        mask = ppgs.model.transformer.mask_from_lengths(
            lengths,
            self.padding
        ).squeeze(dim=1).to(torch.long)

        # Infer
        ppg = self.w2v2fc(inputs, mask, return_dict=True)['logits']

        # Remove <unk> and <pad> tokens
        ppg = ppg[..., :-2]

        # Permute tokens to our ordering
        ppg = ppg.index_select(
            ppg.dim() - 1,
            torch.tensor(ppgs.CHARSIU_PERMUTE).to(ppg.device))

        return ppg.transpose(1, 2)
