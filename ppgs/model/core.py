import torch

import ppgs

###############################################################################
# Model selection
###############################################################################


def Model(type=None):
    type = ppgs.MODEL if type is None else type
    if type == 'convolution':
        print('using convolutional model')
        return ppgs.model.Convolution
    elif type == 'transformer':
        print('using transformer model')
        return lambda: ppgs.model.LengthsWrapperModel(
            [
                torch.nn.Conv1d(ppgs.INPUT_CHANNELS, ppgs.HIDDEN_CHANNELS, kernel_size=ppgs.KERNEL_SIZE, padding="same"),
                ppgs.model.Transformer(),
                torch.nn.Conv1d(ppgs.HIDDEN_CHANNELS, ppgs.OUTPUT_CHANNELS, kernel_size=ppgs.KERNEL_SIZE, padding="same")
            ],
            [False, True, False]
        )
    elif type == 'oldtransformer':
        print('using old transformer model')
        return lambda: ppgs.model.LengthsWrapperModel(
            [
                torch.nn.Conv1d(ppgs.INPUT_CHANNELS, ppgs.HIDDEN_CHANNELS, kernel_size=ppgs.KERNEL_SIZE, padding="same"),
                ppgs.model.OldTransformer(),
                torch.nn.Conv1d(ppgs.HIDDEN_CHANNELS, ppgs.OUTPUT_CHANNELS, kernel_size=ppgs.KERNEL_SIZE, padding="same")
            ],
            [False, True, False]
        )
    elif type == 'Wav2Vec2.0':
        print('using Wav2Vec 2.0 model')
        return lambda: ppgs.model.W2V2()
    raise ValueError('unknown model type:', type)
