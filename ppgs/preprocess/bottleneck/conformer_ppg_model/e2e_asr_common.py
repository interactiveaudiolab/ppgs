#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common functions for ASR."""

import numpy as np


def get_vgg2l_odim(idim, in_channel=3, out_channel=128, downsample=True):
    """Return the output size of the VGG frontend.

    :param in_channel: input channel size
    :param out_channel: output channel size
    :return: output size
    :rtype int
    """
    idim = idim / in_channel
    if downsample:
        idim = np.ceil(np.array(idim, dtype=float) / 2)  # 1st max pooling
        idim = np.ceil(np.array(idim, dtype=float) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels
