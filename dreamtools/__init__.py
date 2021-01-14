from .gams_pandas import *
from .multiindex_plotly import *
import plotly.io as pio

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
pio.orca.config.default_scale = PLOT_SCALE

from .multiindex_plotly.dream_plotly_template import dream_template
pio.templates["dream"] = dream_template
pio.templates.default = "dream"

# Model specific settings
from .multiindex_plotly.makro_settings import *
