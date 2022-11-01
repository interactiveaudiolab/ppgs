import tempfile
from pathlib import Path

import torch

import ppgs


###############################################################################
# Test API
###############################################################################


def test_core(file):
    """Shape test for end-user API"""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / file.with_suffix('.pt').name
        ppgs.from_files_to_files([file], [output])
        assert torch.load(output).shape == (42, 1001)
