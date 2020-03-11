from warnings import warn

from .gams_pandas import Gdx, GamsPandasDatabase
from .gams_pandas import series_from_parameter, series_from_variable, index_names_from_symbol, index_from_symbol
from .gams_pandas import set_symbol_records, merge_symbol_records

import pandas as pd

from .util.pandas_util import unstack_multiseries, merge_multiseries

try:
  import plotly.graph_objects as go
  import plotly.io as pio
  PLOTLY = True
except ImportError:
  warn("Install the plotly package to enable plotting with dream-tools.")
  PLOTLY = False
from math import inf

# Global setting controlling the default position of the time index (-1 = last index is time)
X_AXIS_INDEX = -1

# Time settings, currently used for plotting only
START_YEAR = -inf
END_YEAR = inf

# Global databases
BASELINE = None
SCENARIOS = {}


def time(start, end=None):
  """Set global time settings."""
  global START_YEAR, END_YEAR
  if end is None:
    end = start
  START_YEAR = start
  END_YEAR = end


if PLOTLY:
  def add_trace_pr_column(fig, df, start=None, end=None, go_type=go.Scatter):
    if start is None:
      start = START_YEAR
    if end is None:
      end = END_YEAR
    df = df[start < df.index]
    df = df[df.index <= end]
    for name in df:
      fig.add_trace(go_type(x=df.index, y=df[name], name=name))


  def multiseries_figure(*series, title=None, start=None, end=None):
    fig = go.Figure()
    if title:
      fig.layout = go.Layout(title=go.layout.Title(text=title, xanchor='center'))
    add_trace_pr_column(fig, merge_multiseries(*series, keep_axis_index=X_AXIS_INDEX), start, end)
    return fig


  def plot(*series, start=None, end=None, title=None, renderer=None, file=None):
    fig = multiseries_figure(*series, title=title, start=start, end=end)
    if file is None:
      fig.show(renderer=renderer)
    else:
      fig.write_image(file)

  def _series_plotly(self, start=None, end=None, title=None, **kwargs):
    if title is None:
      title = self.name
    plot(self, start=start, end=end, title=title, **kwargs)

  pd.Series.plot = _series_plotly


  def _data_frame_plotly(self, start=None, end=None, title=None, renderer=None, file=None):
    fig = go.Figure()
    if title:
      fig.layout = go.Layout(title=go.layout.Title(text=title))
    add_trace_pr_column(fig, self, start, end)
    if file is None:
      fig.show(renderer=renderer)
    else:
      fig.write_image(file)

  pd.DataFrame.plot = _data_frame_plotly


def set_renderer(renderer="browser"):
  pio.renderers.default = renderer