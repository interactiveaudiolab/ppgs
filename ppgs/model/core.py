import ppgs


###############################################################################
# Model selection
###############################################################################


def Model(type=ppgs.MODEL):
    if type == 'convolution':
        return ppgs.model.Convolution
    elif type == 'transformer':
        return ppgs.model.Transformer
    raise ValueError('unknown model type:', type)
