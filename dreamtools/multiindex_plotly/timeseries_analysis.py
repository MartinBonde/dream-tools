import dreamtools as dt
import numpy as np
import pandas as pd
import easygui
from IPython.display import display

def time(start, end=None):
  """Set global time settings."""
  if end is None:
    end = start
  dt.START_YEAR = start
  dt.END_YEAR = end

def aggregate_index(index, default_set_aggregations):
  if index.nlevels > 1:
    return tuple(default_set_aggregations.get(k, list(index.levels[i])) for i, k in enumerate(index.names))
  else:
    return default_set_aggregations.get(index.name, list(index))

def aggregate_series(series, default_set_aggregations=None, reference_database=None):
  if default_set_aggregations is None:
    default_set_aggregations = dt.DEFAULT_SET_AGGREGATIONS

  if reference_database is None:
    reference_database = get_reference_database()

  if len(reference_database[series.name].index) == len(series.index):
    return series.loc[aggregate_index(series.index, default_set_aggregations)]
  else:
    return series

def prt(iter_series,
        operator=None,
        start_year=None,
        end_year=None,
        reference_database=None,
        default_set_aggregations=None,
        dec=6,
        max_rows=100,
        max_columns=20,):
  df = to_dataframe(iter_series, operator, start_year, end_year, reference_database, default_set_aggregations)
  df.style.set_properties(**{"text-align": "right", "precision": dec})
  with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):  # more options can be specified also
    display(df)

def plot(iter_series,
         operator=None,
         start_year=None,
         end_year=None,
         reference_database=None,
         default_set_aggregations=None,
         layout={},
         **kwargs):
  df = to_dataframe(iter_series, operator, start_year, end_year, reference_database, default_set_aggregations)
  fig = df.plot(**kwargs)
  layout = {"yaxis": {"title": ""}, "xaxis": {"title": dt.TIME_AXIS_TITLE}, **layout}
  fig.update_layout(layout)
  return fig

def to_dataframe(iter_series, operator=None, start_year=None, end_year=None, reference_database=None, default_set_aggregations=None):
  if isinstance(iter_series, pd.Series):
    iter_series = [iter_series]

  iter_series = [aggregate_series(s, default_set_aggregations, reference_database) for s in iter_series]
  iter_series = compare_to_reference(iter_series, operator, reference_database)

  if start_year is None:
    start_year = dt.START_YEAR
  if end_year is None:
    end_year = dt.END_YEAR
  return merge_multiseries(*iter_series).loc[start_year:end_year]

def get_reference_database():
  if dt.REFERENCE_DATABASE is None:
    dt.REFERENCE_DATABASE = dt.Gdx(easygui.fileopenbox("Select reference gdx file", filetypes=["*.gdx"]))
  return dt.REFERENCE_DATABASE

def compare_to_reference(iter_series, operator, reference_database=None):
  """
  Applies an operator to each series in iter_series, comparing with the series with same name in the reference_database.
  """
  if not operator:
    return [s.copy() for s in iter_series]

  if reference_database is None:
    reference_database = get_reference_database()

  dimension_changed = [reference_database[s.name].index.nlevels != s.index.nlevels for s in iter_series]
  refs = [reference_database[s.name].loc[s.index] if not dimension_changed[i]
          else s.copy() * np.NaN
          for i, s in enumerate(iter_series)]

  if operator in ["q"]:
    return [s / b - 1 for s, b in zip(iter_series, refs)]
  elif operator in ["m"]:
    return [s - b for s, b in zip(iter_series, refs)]
  elif operator in ["s"]:
    for b in refs:
      b.name = "baseline." + b.name
    return [i for pair in zip(iter_series, refs) for i in pair]
  else:
    raise ValueError(f"{operator} is not a valid operator.")

def unstack_multiseries(series, keep_axis_index=-1):
  """
  Return a DataFrame from a Series.
  All levels of the Series are unstacked and concatenated as column names, except <keep_axis_index>.
  """
  if isinstance(series.index, pd.MultiIndex):
    series = series.copy()
    if series.name is None:
      series.name = ""
    series.index = pd.MultiIndex.from_tuples(flatten_keys(series.name, keys, keep_axis_index) for keys in series.index)
    df = series.unstack(0)[series.index.get_level_values(0).unique()]
  else:
    df = pd.DataFrame(series)
  return df

def flatten_keys(name, keys, keep_axis_index):
  keys_str = ','.join(map(str, keys[:keep_axis_index:]))
  flat_name = f"{name}[{keys_str}]"
  return flat_name, keys[keep_axis_index]

def merge_multiseries(*series, keep_axis_index=-1):
  """
  Return a DataFrame from any number of Series, with all levels except <keep_axis_index> concatenated as column names.
  """
  output = pd.DataFrame()
  for s in series:
    df = unstack_multiseries(s, keep_axis_index)
    for c in df.columns:
      output[c] = df[c]
  return output


