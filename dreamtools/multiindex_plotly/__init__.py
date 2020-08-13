PLOTLY = True
try:
  import plotly
except ImportError:
  PLOTLY = False

if PLOTLY:
  import pandas
  pandas.options.plotting.backend = "plotly"
  from .timeseries_analysis import *
  from .plotly_util import *
  from .age_plot import *