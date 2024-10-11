from .gams_pandas import *

__version__ = "2.5.0"

# Global setting controlling the default name of the time index
X_AXIS_NAME = "t"
# Default axis position if no level named dt.X_AXIS_NAME is found
X_AXIS_INDEX = -1 #  (-1 = last index is time)

# Time settings, used for plot and prt functions
START_YEAR = 1983
END_YEAR = 2060

# Age settings, currently used for plotting only
START_AGE = 0
END_AGE = 100

# Global databases
REFERENCE_DATABASE = None

# Ploting
from .multiindex_plotly import *

# Model specific settings
from .multiindex_plotly.makro_settings import DEFAULT_SET_AGGREGATIONS, AGE_AXIS_TITLE, TIME_AXIS_TITLE, YAXIS_TITLE_FROM_OPERATOR

from .utils import *

from .gamY import gamY