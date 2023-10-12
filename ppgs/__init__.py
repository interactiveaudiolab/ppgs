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
from . import notify
from . import checkpoint
from . import data
from . import evaluate
from . import load
from . import model
from . import partition
from . import preprocess
from . import train
from . import write
from . import plot
