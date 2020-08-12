from .gams_pandas import *
from .multiindex_plotly import *

# Global setting controlling the default position of the time index (-1 = last index is time)
X_AXIS_INDEX = -1

# Time settings, used for plot and prt functions
START_YEAR = 1965
END_YEAR = 2100

# Time settings, currently used for plotting only
START_AGE = 0
END_AGE = 100

# Global databases
REFERENCE_DATABASE = None

# Domain specific settings
from .multiindex_plotly import makro_settings as settings
DEFAULT_SET_AGGREGATIONS = settings.DEFAULT_SET_AGGREGATIONS
AGE_AXIS_TITLE = settings.AGE_AXIS_TITLE
TIME_AXIS_TITLE = settings.TIME_AXIS_TITLE
