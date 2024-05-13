import ppgs
import inspect

###############################################################################
# Model selection
###############################################################################


def Model(**kwargs):
    """Initialize PPG model"""
    if ppgs.MODEL == 'convolution':
        init_fn = ppgs.model.Convolution
    elif ppgs.MODEL == 'transformer':
        init_fn = ppgs.model.Transformer
    elif ppgs.MODEL == 'Wav2Vec2.0':
        init_fn = ppgs.model.W2V2
    elif ppgs.MODEL == 'W2V2FC':
        init_fn = ppgs.model.W2V2FC
    else:
        raise ValueError(f'Model {ppgs.MODEL} is not defined')

    sig = inspect.signature(init_fn)
    valid_keys = set(sig.parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return init_fn(**kwargs)
