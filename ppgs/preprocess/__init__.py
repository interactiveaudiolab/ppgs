###############################################################################
# Import Charsiu models from Git submodule
###############################################################################


from pathlib import Path
import importlib.util

try:

    # Import Charsiu
    charsiu_spec = importlib.util.spec_from_file_location(
        'charsiu_models',
        Path(__file__).parent / 'charsiu' / 'src' / 'models.py')
    charsiu_models = importlib.util.module_from_spec(charsiu_spec)
    charsiu_spec.loader.exec_module(charsiu_models)

except (FileNotFoundError, ImportError, ModuleNotFoundError):

    # Continue without Charsiu
    pass


###############################################################################
# Module imports
###############################################################################


from . import bottleneck
from . import w2v2fc
from . import w2v2fb
from . import spectrogram
from . import mel
from . import encodec
from . import dac
from .core import *
