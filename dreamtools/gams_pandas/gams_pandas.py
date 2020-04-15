"""
"""
import os
from collections.abc import Iterable
from six import string_types

import numpy as np
import pandas as pd
import gams
import logging
import builtins
from copy import deepcopy

logger = logging.getLogger(__name__)

def is_iterable(arg):
  return isinstance(arg, Iterable) and not isinstance(arg, string_types)

def map_lowest_level(func, x):
  """Map lowest level of zero or more nested lists."""
  if is_iterable(x):
    return [map_lowest_level(func, i) for i in x]
  else:
    return func(x)

def try_to_int(x):
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


class GamsPandasDatabase:
  """
  GamsPandasDatabase converts sets, parameters, and variables between a GAMS database and Pandas series.
  When as symbol is first retrieved it is converted to a Pandas series and stored in self.series
  Changes to retrieved series are written to the GAMS database on export.
  """
  def __init__(self, database=None, workspace=None):
    if database is None:
      if workspace is None:
        workspace = gams.GamsWorkspace()
      database = workspace.add_database()
    self.database = database
    self.series = {}

  def __getattr__(self, item):
    return self[item]

  def copy(self):
    obj = type(self).__new__(self.__class__)
    obj.__dict__.update(self.__dict__)
    obj.database = self.database.workspace.add_database(source_database=self.database)
    obj.series = deepcopy(self.series)
    return obj

  def merge(self, other, symbol_names=None, inplace=False):
    """
    Merge two GamsPandasDatabases.
    symbol_names: list of symbol names to get from other. If None, all symbols in other are merged.
    """
    if inplace:
      db = self
    else:
      db = self.copy()

    if symbol_names is None:
      symbol_names = other.symbols

    for name in other.sets:
      if name in symbol_names:
        db.add_set_from_series(other[name].texts, other[name].explanatory_text)

    for name in other.variables:
      if name in symbol_names:
        db.add_variable_from_series(other[name], other[name].explanatory_text)

    for name in other.parameters:
      if name in symbol_names:
        db.add_parameter_from_series(other[name], other[name].explanatory_text)

    return db

  @property
  def symbols(self):
    """Dictionary of all symbols in the underlying GAMS database"""
    return {symbol.name: symbol for symbol in self.database}

  @property
  def sets(self):
    """Dictionary of all sets in the underlying GAMS database"""
    return {symbol.name: symbol for symbol in self.database if isinstance(symbol, gams.GamsSet)}

  @property
  def variables(self):
    """Dictionary of all variables in the underlying GAMS database"""
    return {symbol.name: symbol for symbol in self.database if isinstance(symbol, gams.GamsVariable)}

  @property
  def parameters(self):
    """Dictionary of all parameters in the underlying GAMS database"""
    return {symbol.name: symbol for symbol in self.database if isinstance(symbol, gams.GamsParameter)}

  @property
  def equations(self):
    """Dictionary of all equations in the underlying GAMS database"""
    return {symbol.name: symbol for symbol in self.database if isinstance(symbol, gams.GamsEquation)}

  def add_to_builtins(self, *args):
    """Retrieve any number symbol names from the database and add their Pandas representations to the global namespace."""
    for identifier in args:
      setattr(builtins, identifier, self[identifier])

  def get(self, *args):
    """Retrieve any nymber of symbol names and return a list of their Pandas representations."""
    return [self[i] for i in args]

  def add_set_from_series(self, series, explanatory_text=""):
    """Add set symbol to database based on a Pandas series."""
    self.create_set(series.name, series.index, explanatory_text, texts=series)

  def add_parameter_from_series(self, series, explanatory_text="", add_missing_domains=False):
    """Add parameter symbol to database based on a Pandas series."""
    if len(series) == 1:
      df = pd.DataFrame(series)
    else:
      df = series.reset_index()
    self.add_parameter_from_dataframe(series.name, df, explanatory_text, add_missing_domains)

  def add_parameter_from_dataframe(self, identifier, df, explanatory_text="", add_missing_domains=False, value_column_index=-1):
    """Add parameter symbol to database based on a Pandas DataFrame."""
    domains = list(df.columns[:value_column_index:])
    for d in domains:
      if d not in self:
        if add_missing_domains:
          self.add_set_from_series(df[d])
        else:
          raise KeyError(f"'{d}' is not a set in the database. Enable add_missing_domains or add the set to the database manually.")
    self.database.add_parameter_dc(identifier, domains, explanatory_text)
    self.series[identifier] = df.set_index(domains).iloc[:, 0]

  def add_variable_from_series(self, series, explanatory_text="", add_missing_domains=False):
    """Add variable symbol to database based on a Pandas series."""
    if len(series) == 1:
      df = pd.DataFrame(series)
    else:
      df = series.reset_index()
    self.add_variable_from_dataframe(series.name, df, explanatory_text, add_missing_domains)

  def get_index(self, x):
    if x is None or isinstance(x, pd.Index):
      return x
    elif isinstance(x, str):
      return self[x]
    elif len(x) and isinstance(x[0], (pd.Index, tuple, list)):
      multi_index = pd.MultiIndex.from_product(x)
      multi_index.names = [getattr(i, "name", None) for i in x]
      return multi_index
    else:
      return pd.Index(x)

  def get_domains_from_index(self, index, name):
    if hasattr(index, "domains"):
      domains = index.domains
    elif hasattr(index, "name"):
      domains = index.names
    else:
      domains = [index.name]
    return ["*" if i in (None, name) else i for i in domains]

  def create_set(self, name, index, explanatory_text="", texts=None, domains=None):
    """
    Add a new GAMS Set to the database and return an Pandas representation of the Set.
    :param str name: Name of the set
    :param iterable index: Iterable of record keys to be added to the set
    :param str explanatory_text: Explanatory text added to the GAMS set
    :param iterable texts: Iterable of record labels - should match the size of the index parameter.
    :param iterable domains: Names of domains that the set should be defined over
    :return: Pandas Index
    """
    if len(index) and isinstance(index[0], pd.Index):
      multi_index = pd.MultiIndex.from_product(index)
      multi_index.names = [getattr(i, "name", None) for i in index]
      index = multi_index
    elif isinstance(index, pd.Index):
      index = index.copy()
    else:
      index = pd.Index(index)

    index.explanatory_text = explanatory_text

    if texts is None:
      texts = map_lowest_level(str, index)
    index.texts = pd.Series(texts, index=index)
    index.texts.name = index.name

    if domains is None:
      domains = ["*" if i in (None, name) else i for i in self.get_domains_from_index(index, name)]
    index.domains = domains
    index.name = name

    self.database.add_set_dc(index.name, domains, explanatory_text)
    self.series[index.name] = index
    return self[name]

  def create_variable(self, name, index=None, explanatory_text="", data=None, dtype=None, copy=False, add_missing_domains=False):
    if index is not None:
      series = pd.Series(data, self.get_index(index), dtype, name, copy)
      series.explanatory_text = explanatory_text
      self.add_variable_from_series(series, explanatory_text, add_missing_domains)
    elif isinstance(data, pd.DataFrame):
      self.add_variable_from_dataframe(name, data, explanatory_text, add_missing_domains)
    elif isinstance(data, pd.Series):
      self.add_variable_from_series(data, explanatory_text, add_missing_domains)
    else:
      if is_iterable(data) and len(data) and is_iterable(data[0]):
        self.database.add_variable(name, len(data[0]), gams.VarType.Free ,explanatory_text)
      elif is_iterable(data):
        self.database.add_variable(name, 1, gams.VarType.Free, explanatory_text)
      else:
        self.database.add_variable(name, 0, gams.VarType.Free, explanatory_text)
      self[name] = data
    return self[name]

  def create_parameter(self, name, index=None, explanatory_text="", data=None, dtype=None, copy=False, add_missing_domains=False):
    if index is not None:
      series = pd.Series(data, self.get_index(index), dtype, name, copy)
      series.explanatory_text = explanatory_text
      self.add_parameter_from_series(series, explanatory_text, add_missing_domains)
    elif isinstance(data, pd.DataFrame):
      self.add_parameter_from_dataframe(name, data, explanatory_text, add_missing_domains)
    elif isinstance(data, pd.Series):
      self.add_parameter_from_series(data, explanatory_text, add_missing_domains)
    else:
      if is_iterable(data) and len(data) and is_iterable(data[0]):
        self.database.add_parameter(name, len(data[0]), explanatory_text)
      elif is_iterable(data):
        self.database.add_parameter(name, 1, explanatory_text)
      else:
        self.database.add_parameter(name, 0, explanatory_text)
      self[name] = data
    return self[name]

  def add_variable_from_dataframe(self, identifier, df, explanatory_text="", add_missing_domains=False, value_column_index=-1):
    """Add variable symbol to database based on a Pandas DataFrame."""
    domains = list(df.columns[:value_column_index:])
    for d in domains:
      if d not in self:
        if add_missing_domains:
          self.add_set_from_series(df[d])
        else:
          raise KeyError(f"'{d}' is not a set in the database. Enable add_missing_domains or add the set to the database manually.")
    self.database.add_variable_dc(identifier, gams.VarType.Free, domains, explanatory_text)
    self.series[identifier] = df.set_index(domains).iloc[:, 0]

  @staticmethod
  @np.vectorize
  def detuple(t):
    """Returns the iterable unchanged, except if it is a singleton, then the element is returned"""
    if isinstance(t, str):
      return t
    try:
      if len(t) == 1:
        return t[0]
    except TypeError:
      pass
    return t

  def __getitem__(self, item):
    if item not in self.series:
      symbol = self.symbols[item]

      if isinstance(symbol, gams.GamsSet):
        self.series[item] = index_from_symbol(symbol)

      elif isinstance(symbol, gams.GamsVariable):
        if symbol_is_scalar(symbol):
          self.series[item] = symbol.find_record().level if len(symbol) else None
        else:
          self.series[item] = series_from_variable(symbol)

      elif isinstance(symbol, gams.GamsParameter):
        if symbol_is_scalar(symbol):
          self.series[item] = symbol.find_record().value if len(symbol) else None
        else:
          self.series[item] = series_from_parameter(symbol)

      elif isinstance(symbol, gams.GamsEquation):
        return symbol

    return self.series[item]

  def __setitem__(self, name, value):
    if name in self.symbols:
      if not is_iterable(value) and is_iterable(self[name]):  # If assigning a scalar to all records in a series
        value = pd.Series(value, index=self[name].index)
      set_symbol_records(self.symbols[name], value)
      self.series[name] = value
    else:
      if not value.name:
        value.name = name
      self.series[name] = value

  def items(self):
    return self.symbols.items()

  def keys(self):
    return self.symbols.keys()

  def values(self):
    return self.symbols.values()

  def save_series_to_database(self):
    """Save Pandas series to GAMS database"""
    for symbol_name, series in self.series.items():
      set_symbol_records(self.symbols[symbol_name], series)

  def export(self, path):
    """Save changes to database and export database to GDX file."""
    self.save_series_to_database()
    self.database.export(os.path.abspath(path))

  def __iter__(self):
    return iter(self.symbols)

  def __len__(self):
    return len(self.symbols)

  def get_text(self, name):
    """Get explanatory text of GAMS symbol."""
    return self.symbols[name].get_text()


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

def set_symbol_records(symbol, value):
  """Convert Pandas series to records in a GAMS Symbol"""
  if isinstance(symbol, gams.GamsSet):
    if isinstance(value, pd.Index):
      texts = getattr(value, "texts", None)
      value = texts if texts is not None else pd.Series(map(str, value), index=value)
    def add_record(symbol, k, v):
      if not pd.isna(v):
        symbol.add_record(k).text = str(v)
  elif isinstance(symbol, gams.GamsVariable):
    def add_record(symbol, k, v):
      if not pd.isna(v):
        symbol.add_record(k).level = v
  elif isinstance(symbol, gams.GamsParameter):
    def add_record(symbol, k, v):
      if not pd.isna(v):
        symbol.add_record(k).value = v
  else:
    TypeError(f"{type(symbol)} is not (yet) supported by gams_pandas")

  symbol.clear()
  if symbol_is_scalar(symbol):
    add_record(symbol, None, value)
  elif list(value.keys()) == [0]:  # If singleton series
    add_record(symbol, None, value[0])
  else:
    for k, v in value.items():
      add_record(symbol, map_lowest_level(str, k), v)

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

def series_from_variable(symbol, attr="level"):
  """Get a variable symbol from the GAMS database and return an equivalent Pandas series."""
  return pd.Series([getattr(rec, attr) for rec in symbol], index_from_symbol(symbol), name=symbol.name)

def series_from_parameter(symbol):
  """Get a parameter symbol from the GAMS database and return an equivalent Pandas series."""
  return pd.Series([rec.value for rec in symbol], index_from_symbol(symbol), name=symbol.name)

class Gdx(GamsPandasDatabase):
  """Wrapper that opens a GDX file as a GamsPandasDatabase."""
  def __init__(self, file_path, workspace=None):
    if not os.path.splitext(file_path)[1]:
      file_path = file_path + ".gdx"
    self.abs_path = os.path.abspath(file_path)
    logger.info(f"Open GDX file from {self.abs_path}.")
    if workspace is None:
      workspace = gams.GamsWorkspace()
    database = workspace.add_database_from_gdx(self.abs_path)
    super().__init__(database)

  def export(self, path=None, relative_path=True):
    """
    Save changes to database and export database to GDX file.
    If no path is specified, the path of the linked GDX file is used, overwriting the original.
    Use relative_path=True to use export to a path relative to the directory of the originally linked GDX file.
    """
    if path is None:
      path = self.abs_path
    elif relative_path:
      path = os.path.join(os.path.dirname(self.abs_path), path)
    logger.info(f"Export GDX file to {path}.")
    super().export(path)



