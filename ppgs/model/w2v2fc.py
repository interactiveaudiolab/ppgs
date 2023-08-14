import torch
from ppgs.preprocess import charsiu_models
from transformers.utils import logging

import ppgs
from ppgs.model.transformer import mask_from_lengths

logging.set_verbosity_error()

class W2V2FC(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.padding = 400//2 - 160//2

        # Load model
        self.w2v2fc = charsiu_models.Wav2Vec2ForFrameClassification.from_pretrained('charsiu/en_w2v2_fc_10ms')

    def forward(self, input_tensor: torch.Tensor, lengths: torch.Tensor):
        inputs = torch.nn.functional.pad(input_tensor, (self.padding, self.padding)).squeeze(dim=1)
        mask = mask_from_lengths(lengths, self.padding).squeeze(dim=1).to(torch.long)
        ppg: torch.Tensor = self.w2v2fc(inputs, mask, return_dict=True)['logits']
        ppg = ppg[..., :-2] #remove <unk> and <pad>
        ppg = ppg.index_select(ppg.dim()-1, torch.tensor(ppgs.CHARSIU_PERMUTE).to(ppg.device))
        ppg = ppg.transpose(1, 2)
        return ppg