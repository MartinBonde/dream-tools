import dreamtools as dt
import numpy as np
import pandas as pd
from IPython.display import display
from inspect import signature


def get_reference_database(s=None):
  """Get baseline database associated with a GamsPandasDatabase. Defaults to dt.REFERENCE_DATABASE."""
  if isinstance(s, dt.GamsPandasDatabase) and s.reference_database is not None:
    return s.reference_database
  else:
    return dt.REFERENCE_DATABASE

def time(start, end=None):
  """Set global time settings."""
  if end is None:
    end = start
  dt.START_YEAR = start
  dt.END_YEAR = end

def years():
  """Return list of years in current time settings."""
  return list(range(dt.START_YEAR, dt.END_YEAR+1))

def foo(series):
  """For each level in series, index by ["tot"] if it exists, otherwise index by first element."""
  return series.loc[series.index.get_level_values(series.index.names[0]).isin(["tot"])]

def aggregate_series(series, default_set_aggregations=None):
  """Aggregate series according to default_set_aggregations."""
  if default_set_aggregations is None:
    default_set_aggregations = dt.DEFAULT_SET_AGGREGATIONS

  aggregated = series.copy()
  levels = aggregated.index.levels if aggregated.index.nlevels > 1 else [aggregated.index]
  for level in levels:
    aggregated_level = default_set_aggregations.get(level.name, list(level))
    mask = aggregated.index.get_level_values(level.name).isin(aggregated_level)
    if len(aggregated[mask]) > 0:
      aggregated = aggregated.loc[mask]

  return aggregated

def prt(
  iter_series,
  operator=None,
  function=None,
  names=None,
  start_year=None,
  end_year=None,
  reference_database=None,
  default_set_aggregations=None,
  dec=6,
  max_rows=100,
  max_columns=20,
):
  """Print a table of a series or list of series."""
  df = to_dataframe(iter_series, operator, function, names, start_year, end_year, reference_database, default_set_aggregations)
  df.style.set_properties(**{"text-align": "right", "precision": dec})
  with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):  # more options can be specified also
    display(df)

def add_xline(fig, x):
  "Add a vertical line to a plotly figure at x"
  return fig.update_layout(shapes=[dict(
    type='line',
    yref='paper', y0=0, y1=1,
    xref='x', x0=x, x1=x,
    line=dict(
      dash="dash",
    ),
    opacity=0.3,
  )])

def plot(
  iter_series,
  operator=None,
  function=None,
  names=None,
  start_year=None,
  end_year=None,
  reference_database=None,
  default_set_aggregations=None,
  xline=None,
  layout={},
  **kwargs
):
  df = to_dataframe(iter_series, operator, function, names, start_year, end_year, reference_database, default_set_aggregations)
  fig = df.plot(**kwargs)
  layout = {
    "yaxis_title_text": dt.YAXIS_TITLE_FROM_OPERATOR.get(operator, ""),
    "xaxis_title_text": dt.TIME_AXIS_TITLE,
    "legend_title_text": "",
    **layout
  }
  fig.update_layout(layout)
  if xline is not None:
    fig = add_xline(fig, xline)
  return fig

def map_with_baseline(function, data, baselines):
  """Map function to data, passing baselines if function takes two arguments."""
  if len(signature(function).parameters) == 2:
    return map(function, data, baselines)
  else:
    return map(function, data)

def to_dataframe(
  data,
  operator=None,
  function=None,
  names=None,
  start_year=None,
  end_year=None,
  baselines=None,
  default_set_aggregations=None,
):
  if isinstance(data, pd.Series) or isinstance(data, dt.GamsPandasDatabase):
    data = [data]

  if isinstance(data[0], dt.GamsPandasDatabase) and function is None:
    raise ValueError("Must specify function when passing GamsPandasDatabase.")

  if function is None:
    function = lambda x: x

  if baselines is None:
    baselines = [get_reference_database(s) for s in data]

  results = map_with_baseline(function, data, baselines)

  if operator is not None:
    if None in baselines:
      raise ValueError("Cannot compare with baseline if no reference database is set.")
    if isinstance(data[0], pd.Series):
      baseline_series = map(get_baseline_series, data, baselines)
      baseline_results = map_with_baseline(function, baseline_series, baselines)
    else:
      baseline_results = map_with_baseline(function, baselines, baselines)
    results = compare(results, baseline_results, operator)

  aggregated = [aggregate_series(s, default_set_aggregations) for s in results]

  try:
    keep_axis_index = aggregated[0].index.names.index("t")
  except ValueError:
    keep_axis_index = -1 # Default to last index level if no level named "t" is found
 
  df = merge_multiseries(*aggregated, keep_axis_index=keep_axis_index)
  if start_year is None:
    start_year = dt.START_YEAR
  if end_year is None:
    end_year = dt.END_YEAR
  df = df.loc[start_year:end_year]

  if names:
    df.columns = names

  return df

def get_baseline_series(x, b):
  """Lookup the name of series x in the reference database b and return the series from b with the same index as x"""
  if x.name not in b:
    raise KeyError(f"'{x.name}' was not found in the reference database.")

  y = b[x.name]
  if (
    y.index.nlevels != x.index.nlevels
    or y.index.names != x.index.names
  ):
    raise KeyError(f"The dimension of '{x.name}' is different in the reference database. If indexing a single element write [['element']] rather than ['element'] to prevent the series dimension being reduced.")

  return (x - x + y)[x.index] # Adding and subtracting x is a trick to keep the index

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
    keep_axis_name = series.index.names[keep_axis_index]
    if series.name is None:
      series.name = ""
    series.index = pd.MultiIndex.from_tuples(flatten_keys(series.name, keys, keep_axis_index) for keys in series.index)
    df = series.unstack(0)[series.index.get_level_values(0).unique()]
    df.index.name = keep_axis_name
  else:
    df = pd.DataFrame(series)
  return df

def flatten_keys(name, keys, keep_axis_index):
  keys_str = ','.join(map(str, [keys[i] for i, _ in enumerate(keys) if i != keep_axis_index]))
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
