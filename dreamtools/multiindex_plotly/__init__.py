import plotly.io as pio
import pandas
pandas.options.plotting.backend = "plotly"
from .timeseries_analysis import *
from .reporting import *
from .age_plot import *

from .dream_plotly_template import create_dream_template
pio.templates["dream"] = create_dream_template(trace_count=6, items_per_row=2)
pio.templates.default = "dream"

from .dream_plotly_template import dream_colors_rgb, small_figure_layout, large_figure_layout

