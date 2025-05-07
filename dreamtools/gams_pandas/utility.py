from typing import Iterable

import gams.transfer as gt
import pandas as pd
from six import string_types

def all_na(x):
  """Returns bool of whether a series or scalar consists of all NAs"""
  if is_iterable(x):
    return all(pd.isna(x))
  else:
    return pd.isna(x)
  
def domains_as_strings(symbol):
    if hasattr(symbol, "domain"):
        return [getattr(d, "name", d) for d in symbol.domain]
        #return symbol.domain
    elif hasattr(symbol, "domains"):
        return symbol.domains
    else:
        raise AttributeError(f"Symbol {symbol.name} has neither 'domain' nor 'domains' attribute")
    
def index_names_from_symbol(symbol):
  """
  Return the domain names of a GAMS symbol,
  except ['*'] cases are replaced by the name of the symbol
  and ['*',..,'*'] cases are replaced with ['index_0',..'index_n']
  """
  index_names = list(domains_as_strings(symbol))
  if index_names == ["*"]:
    return [symbol.name]
  if index_names.count("*") > 1:
    for i, name in enumerate(index_names):
      if name == "*":
        index_names[i] = f"index_{i}"
  return index_names


def better_index_from_symbol(symbol):
  """Return a Pandas Index based on the records and domain names of a GAMS symbol."""
  records=symbol.records
  if len(domains_as_strings(symbol)) > 1:
    keys=[tuple(row) for row in records.iloc[:,len(symbol.domains)].values]
    index = list(map_to_index_where_possible(pd.MultiIndex.from_tuples(keys, names=index_names_from_symbol(symbol))))
    index.name = symbol.name
  elif len(domains_as_strings(symbol)) == 1:
    keys=records.iloc[:,0].tolist()
    index = pd.Index(map_to_int_where_possible(pd.Index(keys, name=index_names_from_symbol(symbol)[0])))
    index.name=domains_as_strings(symbol)[0]
  else:
    return None  
  if isinstance(symbol, gt.Set):
    index.domains = domains_as_strings(symbol)
    idx = pd.MultiIndex.from_frame(symbol.records[symbol.domain_labels])
    '''
    below line fixes the issue of vars/params having domains of sets as names rather than sets,
    but as a consequence, when reading a set, it's header is not its domain causing an assertion in gdx_read to fail.
    '''
    index.names=[symbol.name]
    if 'element_text' in symbol.records.columns:
        index.texts = pd.Series(symbol.records['element_text'].values, index=idx, name=symbol.name)
    else:
        index.texts = pd.Series([None] * len(symbol.records), index=idx, name=symbol.name)
  
  return index  


def symbol_is_scalar(symbol):
  return not domains_as_strings(symbol)


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

def safe_set_records(symbol, value):
    '''This function ensures domains are strings + categorigcal and values are simple (floats) and categorical, required for export with gams.transfer'''
    if symbol_is_scalar(symbol):
        symbol.records = pd.DataFrame([{'level': value}])
        return

    if isinstance(value, pd.Series):
        value = value.reset_index()
    elif isinstance(value, (list, tuple)) and isinstance(value[0], dict):
        value = pd.DataFrame(value)
    elif not isinstance(value, pd.DataFrame):
        raise ValueError(f"Cannot assign {type(value)} to non-scalar symbol {symbol.name}")

    expected_cols = [d.name if hasattr(d, "name") else d for d in symbol.domain] + ['level']
    missing_cols = set(expected_cols) - set(value.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} for symbol {symbol.name}")
    #ensure domains are strings + categorical
    for d in symbol.domain:
        dname = d.name if hasattr(d, "name") else d
        if dname in value.columns:
            value[dname] = value[dname].astype(str).astype("category")
    #ensure values are float
    data_columns=set(value.columns)-{d.name if hasattr(d,'name') else d for d in symbol.domain}
    for col in data_columns:
       value[col]=value[col].astype(float)
    symbol.records = value