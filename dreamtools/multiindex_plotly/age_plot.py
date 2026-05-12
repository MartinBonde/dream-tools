import plotly.graph_objects as go
import dreamtools as dt
import pandas as pd
from pandas import IndexSlice

from .dream_plotly_template import FONT_FAMILY
from .timeseries_analysis import _DataFrame

def age_figure_3d(series,
                  start_year=None,
                  end_year=None,
                  start_age=None,
                  end_age=None,
                  title="",
                  ztitle="",
                  showscale=False,
                  **kwargs):
  if start_year is None:
    start_year = max(dt.START_YEAR, min(series.index.levels[-1]))
  if end_year is None:
    end_year = min(dt.END_YEAR, max(series.index.levels[-1]))
  if start_age is None:
    start_age = dt.START_AGE
  if end_age is None:
    end_age = dt.END_AGE

  age = list(range(start_age, end_age + 1))
  time = list(range(start_year, end_year + 1))
  value = series.loc[age, time].unstack().values
  surface = go.Surface(x=time, y=age, z=value, showscale=showscale, **kwargs)
  return go.Figure(
    surface,
    layout={
      "template": "dream",
      "font": dict(family=FONT_FAMILY),
      "scene": {
        "xaxis": {"title": dt.time_axis_title(), "autorange": "reversed"},
        "yaxis": {"title": dt.age_axis_title(), "autorange": "reversed"},
        "zaxis": {"title": ztitle},
      },
      "title": {"text": title, 'x': 0.5, "y": 0.925}
    },
  )

def dummy_function(x):
  return x

def age_figure_2d(iter_series,
                  operator=None,
                  years=None,
                  start_age=None, end_age=None,
                  reference_database=None,
                  names=None,
                  function=dummy_function,
                  layout=None,
                  xline=None,
                  vertical_legend=False,
                  horizontal_yaxis_title=False,
                  figure_size="document_large",
                  legend_label_width="auto",
                  colored_legend=True,
                  alternating_dash=None,
                  **kwargs
                  ):
  if isinstance(iter_series, pd.Series):
    iter_series = [iter_series]
  if years is None:
    years = list(range(dt.START_YEAR, dt.END_YEAR+1))
    if max(len(s.loc[:,years].index.unique(level=-1)) for s in iter_series) > 5:
      years = list(range(dt.START_YEAR, dt.END_YEAR + 1, 5))
  if start_age is None:
    start_age = dt.START_AGE
  if end_age is None:
    end_age = dt.END_AGE
  iter_series = [function(series.sort_index().loc[IndexSlice[start_age:end_age, years]]) for series in iter_series]
  if operator:
    if reference_database is None:
      reference_database = dt.get_reference_database()
    refs = [function(reference_database[series.name].sort_index().loc[series.index]) for series in iter_series]
    iter_series = dt.compare(iter_series, refs, operator)
  if names is not None:
    for name, series in zip(names, iter_series):
      series.name = name
  df = pd.DataFrame()
  for series in iter_series:
    series_df = series.unstack()
    if len(series_df.columns) == 1:
      df[f"{series.name}"] = series_df[series_df.columns[0]]
    else:
      for col in series_df:
        df[f"{series.name}[{col}]"] = series_df[col]
  df = _DataFrame(df)
  df.layout = {
    "xaxis_title_text": dt.age_axis_title(),
    "yaxis_title_text": dt.yaxis_title_from_operator(operator),
    "legend_title_text": "",
  }
  merged_layout = {**(layout or {}), **kwargs}
  return df.plot(
    operator=operator,
    layout=merged_layout,
    xline=xline,
    vertical_legend=vertical_legend,
    horizontal_yaxis_title=horizontal_yaxis_title,
    figure_size=figure_size,
    legend_label_width=legend_label_width,
    colored_legend=colored_legend,
    alternating_dash=alternating_dash,
  )