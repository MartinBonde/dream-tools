import pandas as pd


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
    df = series.unstack(0)
  else:
    df = pd.DataFrame(series)
  df.index = pd.to_numeric(df.index)
  df.index.name = "t"
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