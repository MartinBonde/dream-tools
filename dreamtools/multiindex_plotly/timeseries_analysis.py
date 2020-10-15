import dreamtools as dt
import numpy as np
import pandas as pd
from IPython.display import display

def time(start, end=None):
  """Set global time settings."""
  if end is None:
    end = start
  dt.START_YEAR = start
  dt.END_YEAR = end

def years():
  return list(range(dt.START_YEAR, dt.END_YEAR+1))

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

  if series.name in reference_database:
    if len(reference_database[series.name].index) == len(series.index):
      return series.loc[aggregate_index(series.index, default_set_aggregations)]
  elif series.index.name in reference_database:
    if len(reference_database[series.index.name]) == len(series.index):
      return series.loc[aggregate_index(series.index, default_set_aggregations)]

  return series

def prt(iter_series,
        operator=None,
        start_year=None,
        end_year=None,
        reference_database=None,
        default_set_aggregations=None,
        function=None,
        dec=6,
        max_rows=100,
        max_columns=20,):
  df = to_dataframe(iter_series, operator, start_year, end_year, reference_database, default_set_aggregations, function)
  df.style.set_properties(**{"text-align": "right", "precision": dec})
  with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):  # more options can be specified also
    display(df)

def plot(iter_series,
         operator=None,
         start_year=None,
         end_year=None,
         reference_database=None,
         default_set_aggregations=None,
         function=None,
         names=None,
         layout={},
         **kwargs):
  df = to_dataframe(iter_series, operator, start_year, end_year, reference_database, default_set_aggregations, function)
  if names:
    df.columns = names
  fig = df.plot(**kwargs)
  layout = {
    "yaxis_title_text": dt.YAXIS_TITLE_FROM_OPERATOR.get(operator, ""),
    "xaxis_title_text": dt.TIME_AXIS_TITLE,
    "legend_title_text": "",
    **layout
  }
  fig.update_layout(layout)
  return fig

def to_dataframe(iter_series,
                 operator=None,
                 start_year=None,
                 end_year=None,
                 reference_database=None,
                 default_set_aggregations=None,
                 function=None):
  if isinstance(iter_series, pd.Series):
    iter_series = [iter_series]

  if function is None:
    function = lambda x: x

  iter_series = [function(aggregate_series(s, default_set_aggregations, reference_database))
                 for s in iter_series]
  if operator:
    if reference_database is None:
      reference_database = get_reference_database()
    refs = [function((s - s + reference_database[s.name])[s.index])
            if not dimension_changed(s, reference_database)
            else s * np.NaN
            for i, s in enumerate(iter_series)]
    iter_series = compare(iter_series, refs, operator)

  if start_year is None:
    start_year = dt.START_YEAR
  if end_year is None:
    end_year = dt.END_YEAR

  return merge_multiseries(*iter_series).loc[start_year:end_year]

def dimension_changed(series, reference_database):
  if series.name in reference_database:
    is_changed = reference_database[series.name].index.nlevels != series.index.nlevels
    if is_changed:
      Warning(
        f"The dimension of '{series.name}' is different in the reference database. If indexing a single element write [['element']] rather than ['element'] to prevent the series dimension being reduced.")
    return is_changed
  else:
    Warning(f"'{series.name}' was not found in the reference database.")
    return True


def get_reference_database():
  if dt.REFERENCE_DATABASE is None:
    import easygui
    dt.REFERENCE_DATABASE = dt.Gdx(easygui.fileopenbox("Select reference gdx file", filetypes=["*.gdx"]))
  return dt.REFERENCE_DATABASE

def compare(iter_series, refs, operator):
  """
  Applies an operator to each pair in zip(iter_series, refs)
  """
  if operator in ["q"]:
    return [s / b - 1 for s, b in zip(iter_series, refs)]
  elif operator in ["pq"]:
    return [(s / b - 1)*100 for s, b in zip(iter_series, refs)]
  elif operator in ["m"]:
    return [s - b for s, b in zip(iter_series, refs)]
  elif operator in ["s"]:
    for b in refs:
      b.name = "baseline." + b.name
    return [i for pair in zip(iter_series, refs) for i in pair]
  elif operator in ["p"]:
    return [s / lag(s) * 100 - 100 for s in iter_series]
  elif operator in ["d"]:
    return [s - lag(s) for s in iter_series]
  elif operator in ["i"]:
    return [index(s) for s in iter_series]
  elif operator in ["log"]:
    return [np.log(s) for s in iter_series]
  elif operator in ["rlog"]:
    return [np.log(s) for s in refs]
  elif operator in ["dlog"]:
    return [np.log(s) - np.log(lag(s)) for s in iter_series]
  else:
    raise ValueError(f"{operator} is not a valid operator.")

def index(series, level=-1, element=None):
  """Return series divided by the value with the index element. E.g. set a series to 1 in a base year."""
  if element is None:
    element = dt.START_YEAR
  if len(series.index.names) > 1:
    return series / series.xs(element, level=level)
  else:
    return series / series[element]

def lag(series, periods=1, lag_axis_index=-1):
  if len(series.index.names) > 1:
    return series.groupby(level=series.index.names[:lag_axis_index:]).shift(periods)
  else:
    return series.shift(periods)

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
      new_name = c if (c != "0") else ""
      iter = 0
      while new_name in output:
        iter += 1
        new_name = f"{c}{iter}"
      output[new_name] = df[c]
  return output


