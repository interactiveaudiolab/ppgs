import ppgs


###############################################################################
# Model selection
###############################################################################


def Model():
    """Initialize PPG model"""
    if ppgs.MODEL == 'convolution':
        return ppgs.model.Convolution()
    elif ppgs.MODEL == 'transformer':
        return ppgs.model.Transformer()
    elif ppgs.MODEL == 'Wav2Vec2.0':
        return ppgs.model.W2V2()
    elif ppgs.MODEL == 'W2V2FC':
        return ppgs.model.W2V2FC()
    raise ValueError(f'Model {ppgs.MODEL} is not defined')
