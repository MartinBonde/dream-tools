from typing import Iterable

import gams
import pandas as pd
from six import string_types

def all_na(x):
  """Returns bool of whether a series or scalar consists of all NAs"""
  if is_iterable(x):
    return all(pd.isna(x))
  else:
    return pd.isna(x)


def index_names_from_symbol(symbol):
  """
  Return the domain names of a GAMS symbol,
  except ['*'] cases are replaced by the name of the symbol
  and ['*',..,'*'] cases are replaced with ['index_0',..'index_n']
  """
  index_names = list(symbol.domains_as_strings)
  if index_names == ["*"]:
    return [symbol.name]
  if index_names.count("*") > 1:
    for i, name in enumerate(index_names):
      if name == "*":
        index_names[i] = f"index_{i}"
  return index_names


def index_from_symbol(symbol):
  """Return a Pandas Index based on the records and domain names of a GAMS symbol."""
  if len(symbol.domains_as_strings) > 1:
    keys = map_to_int_where_possible([rec.keys for rec in symbol])
    index = pd.MultiIndex.from_tuples(keys, names=index_names_from_symbol(symbol))
    index.name = symbol.name
  elif len(symbol.domains_as_strings) == 1:
    keys = map_to_int_where_possible([rec.keys[0] for rec in symbol])
    index = pd.Index(keys, name=index_names_from_symbol(symbol)[0])
  else:
    return None
  if isinstance(symbol, gams.GamsSet):
    index.text = symbol.text
    index.domains = symbol.domains_as_strings
    index.texts = pd.Series([rec.text for rec in symbol], index, name=symbol.name)
  return index


def symbol_is_scalar(symbol):
  return not symbol.domains_as_strings


def is_iterable(arg):
  return isinstance(arg, Iterable) and not isinstance(arg, string_types)


def map_lowest_level(func, x):
  """Map lowest level of zero or more nested lists."""
  if is_iterable(x):
    return [map_lowest_level(func, i) for i in x]
  else:
    return func(x)


def try_to_int(x):
  """Cast input to int if possible, else return input unchanged."""
  try:
    if str(int(x)) == str(x):
      return int(x)
    else:
      return x
  except ValueError:
    return x


def map_to_int_where_possible(iter):
  """Returns an iterable where each element is converted to an integer if possible for that element."""
  return map_lowest_level(try_to_int, iter)


def merge_symbol_records(series, symbol):
  """Convert Pandas series to records in a GAMS Symbol"""
  if isinstance(symbol, gams.GamsSet):
    attr = "text"
  elif isinstance(symbol, gams.GamsVariable):
    attr = "level"
  elif isinstance(symbol, gams.GamsParameter):
    attr = "value"
  for k, v in series.items():
    setattr(symbol.merge_record(k), attr, v)


def fill_missing_combinations(series, sets_database=None, fill_value=pd.NA):
  """
  Return copy of series with all combinations filled with fill_value.
  If a database is supplied we look up the sets in the database, otherwise we only fill set elements already in use.
  """
  sets = [i.unique() for i in series.index.levels[:]]
  if sets_database is not None:
    for i, set_name in enumerate(series.index.names):
      if set_name in sets_database:
        sets[i] = sets_database[set_name]
  all_combinations = pd.MultiIndex.from_product(sets)
  return series.reindex(all_combinations, fill_value=fill_value)
