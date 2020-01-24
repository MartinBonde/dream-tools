from .gams_pandas import Gdx, GamsPandasDatabase
from .gams_pandas import series_from_parameter, series_from_set, series_from_variable, index_names_from_symbol, index_from_symbol
from .gams_pandas import set_symbol_records, merge_symbol_records
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from math import inf

# Global setting controlling the default position of the time index (-1 = last index is time)
X_AXIS_INDEX = -1

# Time settings, currently used for plotting only
START_YEAR = -inf
END_YEAR = inf


def time(start, end=None):
  """Set global time settings."""
  global START_YEAR, END_YEAR
  if end is None:
    end = start
  START_YEAR = start
  END_YEAR = end


def unstack_multiseries(series):
  if isinstance(series.index, pd.MultiIndex):
    df = series.copy()
    if series.name is None:
      series.name = ""
    df.index = pd.MultiIndex.from_tuples((f"{series.name}[{','.join(i[:X_AXIS_INDEX:])}]", i[X_AXIS_INDEX]) for i in df.index)
    df = df.unstack(0)
  else:
    df = pd.DataFrame(series)
  df.index = pd.to_numeric(df.index)
  df.index.name = "t"
  return df


def merge_multiseries(*series):
  output = pd.DataFrame()
  for s in series:
    df = unstack_multiseries(s)
    for c in df.columns:
      output[c] = df[c]
  return output


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
  add_trace_pr_column(fig, merge_multiseries(*series), start, end)
  return fig


def plot(*series, title=None, start=None, end=None, renderer=None, file=None):
  fig = multiseries_figure(*series, title=title, start=start, end=end)
  if file is None:
    fig.show(renderer=renderer)
  else:
    fig.write_image(file)


class GamsPandasDatabase(GamsPandasDatabase):
  def plot(self, *identifiers, **kwargs):
    plot(*[self[id] for id in identifiers], **kwargs)


class Gdx(GamsPandasDatabase, Gdx):
  pass


def _series_plot(self, start=None, end=None, title=None, **kwargs):
  if title is None:
    title = self.name
  plot(self, start=start, end=end, title=title, **kwargs)
pd.Series.plot = _series_plot


def _data_frame_plot(self, start=None, end=None, title=None, renderer=None, file=None):
  fig = go.Figure()
  if title:
    fig.layout = go.Layout(title=go.layout.Title(text=title))
  add_trace_pr_column(fig, self, start, end)
  if file is None:
    fig.show(renderer=renderer)
  else:
    fig.write_image(file)
pd.DataFrame.plot = _data_frame_plot


def set_renderer(renderer="browser"):
  pio.renderers.default = renderer