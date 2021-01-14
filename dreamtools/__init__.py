from .gams_pandas import *

# Global setting controlling the default position of the time index (-1 = last index is time)
X_AXIS_INDEX = -1

# Time settings, used for plot and prt functions
START_YEAR = 1965
END_YEAR = 2100

# Age settings, currently used for plotting only
START_AGE = 0
END_AGE = 100

# Global databases
REFERENCE_DATABASE = None

# Plotly settings
LARGE_PLOT_WIDTH = 1830
PLOT_HEIGHT = 592
SMALL_PLOT_WIDTH = 897
PLOT_SCALE = 3
from .multiindex_plotly import *

# Model specific settings
from .multiindex_plotly.makro_settings import *
