from pathlib import Path

import pytest

import ppgs


TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Pytest fixtures
###############################################################################


@pytest.fixture(scope='session')
def dataset():
    """Preload the dataset"""
    return ppgs.data.Dataset('arctic', 'valid')


@pytest.fixture(scope='session')
def model():
    """Preload the model"""
    return ppgs.model.Model()
