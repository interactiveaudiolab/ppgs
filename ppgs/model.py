"""model.py - model definition"""


import torch


###############################################################################
# Model
###############################################################################


class Model(torch.nn.Module):
    """Model definition"""

    # TODO - add hyperparameters as input args
    def __init__(self):
        super().__init__()

        # TODO - define model
        raise NotImplementedError

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self):
        """Perform model inference"""
        # TODO - define model arguments and implement forward pass
        raise NotImplementedError
