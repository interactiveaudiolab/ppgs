import ppgs
import torch

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
        return lambda: ppgs.model.WrapperModel(
            [ppgs.model.Transformer(), torch.nn.Conv1d(ppgs.INPUT_CHANNELS, len(ppgs.PHONEME_LIST), kernel_size=ppgs.KERNEL_SIZE, padding="same")],
            [2, 1]
        )
    raise ValueError('unknown model type:', type)
