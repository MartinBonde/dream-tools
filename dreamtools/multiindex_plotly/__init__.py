try:
  import plotly
  PLOTLY = True
except ImportError:
  PLOTLY = False

if PLOTLY:
  import pandas
  pandas.options.plotting.backend = "plotly"
  from .timeseries_analysis import *
  from .reporting import *
  from .age_plot import *

  import plotly.io as pio
  from .dream_plotly_template import dream_template
  pio.templates["dream"] = dream_template
  pio.templates.default = "dream"

  from .dream_plotly_template import dream_colors_rgb, small_figure_layout, large_figure_layout
