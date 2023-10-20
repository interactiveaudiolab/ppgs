###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('ppgs', defaults)

# Import configuration parameters
from .config.defaults import *
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .phonemes import *
from .core import *
from .model import Model
from .train import loss, train
from . import data
from . import edit
from . import evaluate
from . import load
from . import model
from . import partition
from . import preprocess
from . import plot
