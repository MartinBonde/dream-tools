"""
"""
import os
import numpy as np
import pandas as pd
import gams
import logging
import builtins
from copy import deepcopy

logger = logging.getLogger(__name__)


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

  def add_set(self, identifier, dimension=1, explanatory_text=""):
    """Add set symbol to database."""
    self.database.add_set(identifier, dimension, explanatory_text)

  def add_set_dc(self, identifier, domains, explanatory_text=""):
    """Add set symbol to database using domain information."""
    self.database.add_set_dc(identifier, [self.symbols[i] for i in domains], explanatory_text)

  def add_parameter(self, identifier, dimension, explanatory_text=""):
    """Add parameter symbol to database."""
    self.database.add_parameter(identifier, dimension, explanatory_text)

  def add_parameter_dc(self, identifier, domains, explanatory_text=""):
    """Add parameter symbol to database using domain information."""
    self.database.add_parameter_dc(identifier, [self.symbols[i] for i in domains], explanatory_text)
    self.series[identifier] = pd.Series(
      index=pd.MultiIndex.from_product([self[i].index for i in domains], names=[self[i].name for i in domains]),
      name=identifier
    ).sort_index()

  def add_variable(self, identifier, dimension, explanatory_text="", vartype=gams.VarType.Free):
    """Add variable symbol to database."""
    self.database.add_variable(identifier, dimension, vartype, explanatory_text)

  def add_variable_dc(self, identifier, domain_names, explanatory_text="", vartype=gams.VarType.Free):
    """Add variable symbol to database using domain information."""
    self.database.add_variable_dc(identifier, vartype, [self.symbols[i] for i in domain_names], explanatory_text)
    self.series[identifier] = pd.Series(
      name=identifier,
      index=pd.MultiIndex.from_product([self.get_set_index(i) for i in domain_names], names=domain_names)
    ).sort_index()

  def add_to_builtins(self, *args):
    """Retrieve any number symbol names from the database and add their Pandas representations to the global namespace."""
    for identifier in args:
      setattr(builtins, identifier, self[identifier])

  def get(self, *args):
    """Retrieve any nymber of symbol names and return a list of their Pandas representations."""
    return [self[i] for i in args]

  def get_set_index(self, name):
    """Return Pandas index based on a retrieved GAMS set."""
    return index_from_symbol(self.symbols[name])

  def add_set_from_series(self, series, explanatory_text=""):
    """Add set symbol to database based on a Pandas series."""
    if not isinstance(series.values[0], str):
      raise TypeError("To convert a Pandas series to a GAMS set, the series must contain strings (element texts).")
    self.database.add_set(series.name, len(series.index.names), explanatory_text)
    if isinstance(series.index, pd.RangeIndex):
      series = series.copy()
      series.index = series.index.astype(str)
    self[series.name] = series

  def add_set_from_index(self, index, explanatory_text=""):
    """Add set symbol to database based on a Pandas index."""
    self.database.add_set(index.name, len(index.names), explanatory_text)
    self[index.name] = pd.Series([str(i) for i in index], index=index, name=index.name)

  def add_parameter_from_series(self, series, explanatory_text="", add_missing_domains=False):
    """Add parameter symbol to database based on a Pandas series."""
    self.add_parameter_from_dataframe(series.name, series.reset_index(), explanatory_text, add_missing_domains)

  def add_parameter_from_dataframe(self, identifier, df, explanatory_text="", add_missing_domains=False, value_column_index=-1):
    """Add parameter symbol to database based on a Pandas DataFrame."""
    domains = df.columns[:value_column_index:]
    for d in domains:
      if d not in self:
        if add_missing_domains:
          self.add_set_from_series(df[d])
        else:
          raise KeyError(f"'{d}' is not a set in the database. Enable add_missing_domains or add the set to the database manually.")
    self.add_parameter_dc(identifier, domains, explanatory_text)
    self[identifier] = df.set_index(list(domains)).iloc[:, 0]

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
        self.series[item] = list_from_set(symbol)
        # self.series[item] = series_from_set(symbol)
      elif isinstance(symbol, gams.GamsVariable):
        self.series[item] = series_from_variable(symbol)
      elif isinstance(symbol, gams.GamsParameter):
        self.series[item] = series_from_parameter(symbol)
      elif isinstance(symbol, gams.GamsEquation):
        return symbol
    return self.series[item]

  def __setitem__(self, key, value):
    try:
      set_symbol_records(value, self.symbols[key])
      self.series[key] = value
    except AttributeError:
      self.series[key] = pd.Series(value, index=self.series[key].index)

  def items(self):
    return self.symbols.items()

  def keys(self):
    return self.symbols.keys()

  def values(self):
    return self.symbols.values()

  def save_series_to_database(self):
    """Save Pandas series to GAMS database"""
    for symbol_name, series in self.series.items():
      set_symbol_records(series, self.symbols[symbol_name])

  def export(self, path):
    """Save changes to database and export database to GDX file."""
    self.save_series_to_database()
    self.database.export(path)

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


def set_symbol_records(series, symbol):
  """Convert Pandas series to records in a GAMS Symbol"""
  if isinstance(symbol, gams.GamsSet):
    attr = "text"
  elif isinstance(symbol, gams.GamsVariable):
    attr = "level"
  elif isinstance(symbol, gams.GamsParameter):
    attr = "value"
  symbol.clear()
  for k, v in series.items():
    setattr(symbol.add_record(k), attr, v)


def index_names_from_symbol(symbol):
  """Return the domain names of a GAMS symbol with '*' replaced by 'index_{i}'"""
  index_names = []
  for i, name in enumerate(symbol.domains_as_strings):
    if name != "*":
      index_names.append(name)
    else:
      index_names.append(f"index_{i}")
  return index_names


def index_from_symbol(symbol):
  """Return a MultiIndex based on the records and domain names of a GAMS symbol."""
  if len(symbol.domains) > 1:
    return pd.MultiIndex.from_tuples([rec.keys for rec in symbol], names=index_names_from_symbol(symbol))
  elif len(symbol.domains) == 1:
    return pd.Index([rec.keys[0] for rec in symbol], name=index_names_from_symbol(symbol)[0])
  else:
    return None

def series_from_variable(symbol, attr="level"):
  """Get a variable symbol from the GAMS database and return an equivalent Pandas series."""
  s = pd.Series([getattr(rec, attr) for rec in symbol], index_from_symbol(symbol), name=symbol.name)
  return s.sort_index()


def series_from_parameter(symbol):
  """Get a parameter symbol from the GAMS database and return an equivalent Pandas series."""
  s = pd.Series([rec.value for rec in symbol], index_from_symbol(symbol), name=symbol.name)
  return s.sort_index()


def series_from_set(symbol):
  """Get a set symbol from the GAMS database and return an equivalent Pandas series."""
  s = pd.Series([rec.text for rec in symbol], index_from_symbol(symbol), name=symbol.name)
  return s.sort_index()


def list_from_set(symbol):
  """Get a set symbol from the GAMS database and return a list of indices."""
  return list(index_from_symbol(symbol).sort_values())


class Gdx(GamsPandasDatabase):
  """Wrapper that opens a GDX file as a GamsPandasDatabase."""
  def __init__(self, file_path, workspace=None):
    self.abs_path = os.path.abspath(file_path)
    logger.info(f"Open GDX file from {self.abs_path}.")
    if workspace is None:
      workspace = gams.GamsWorkspace()
    database = workspace.add_database_from_gdx(self.abs_path)
    super().__init__(database)

  def export(self, path=None, relative_path=False):
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


# ----------------------------------------------------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------------------------------------------------
@np.vectorize
def approximately_equal(x, y, ndigits=0):
  return round(x, ndigits) == round(y, ndigits)


def test_gdx_read():
  assert approximately_equal(Gdx("test.gdx")["qY"]["byg", "2010"], 191)
  assert approximately_equal(Gdx("test.gdx")["qI_s"]["IB", "fre", "2010"], 4.43)  # "IB" should be changed to "iB" in the GDX file.
  assert approximately_equal(Gdx("test.gdx")["eCx"], 1)


def test_add_symbols():
  db = GamsPandasDatabase()
  db.add_set("year", 1, "Years")
  db.add_set("sector", 1, "Sectors")
  db["year"] = pd.Series([str(i) for i in range(2010, 2026)], index=[str(i) for i in range(2010, 2026)], name="t")
  db["sector"] = pd.Series(["Services", "Industrial production"], index=["ser", "goo"], name="s")
  db.add_parameter_dc("par", ["year", "sector"], explanatory_text="Parameter added from Python.")
  db.add_variable_dc("var", ["year", "sector"], explanatory_text="Variable added from Python.")
  db["par"] = 1
  db["var"] = 1
  assert all(approximately_equal(db["par"], db["var"]))
  db["var"]["2020":] = 2
  assert any(approximately_equal(db["par"], db["var"]))
  assert not all(approximately_equal(db["par"], db["var"]))
  db["var"] = 2
  assert not any(approximately_equal(db["par"], db["var"]))
  assert db.get_text("var") == "Variable added from Python."


def test_add_set_from_index():
  db = GamsPandasDatabase()
  t = pd.Index([str(i) for i in range(2010, 2026)], name="t")
  db.add_set_from_index(t)
  assert db["t"].name == "t"
  assert all(db["t"].index == t)

  s = pd.Index(["services", "goods"], name="s")
  st = pd.MultiIndex.from_product([s,t], names=["s", "t"])
  st.name = "st"
  db.add_set_from_index(st)
  assert db["t"].name == "t"
  assert all(db["t"].index == t)


def test_add_set_from_series():
  db = GamsPandasDatabase()
  t = pd.Index([str(i) for i in range(2010, 2026)], name="t")
  t = pd.Series(t, index=t, name="t")
  db.add_set_from_series(t)
  assert db["t"].name == "t"
  assert all(db["t"] == t)

  s = pd.Index(["services", "goods"], name="s")
  st = pd.MultiIndex.from_product([s,t], names=["s", "t"])
  st = pd.Series([str(i) for i in st], index=st, name="st")
  db.add_set_from_series(st)
  assert db["st"].name == "st"
  assert all(db["st"] == st)


def test_add_parameter_from_dataframe():
  db = GamsPandasDatabase()
  df = pd.DataFrame()
  df["t"] = [str(i) for i in range(2010, 2026)]
  df["value"] = 1.3
  db.add_parameter_from_dataframe("par", df, add_missing_domains=True)
  assert all(db["par"] == 1.3)
  assert len(db["par"]) == 16


def test_multiply_added():
  db = GamsPandasDatabase()
  df = pd.DataFrame([
      ["2010", "ser", 3],
      ["2010", "goo", 2],
      ["2020", "ser", 6],
      ["2020", "goo", 4],
    ], columns=["t", "s", "value"])
  db.add_parameter_from_dataframe("q", df, add_missing_domains=True)

  df = pd.DataFrame([
      ["2010", 1],
      ["2020", 1.2],
    ], columns=["t", "value"])
  db.add_parameter_from_dataframe("p", df, add_missing_domains=True)

  v = db["p"] * db["q"]
  v.name = "v"
  db.add_parameter_from_series(v)

  assert db["v"]['2020', "goo"] == 4.8


def test_add_parameter_from_series():
  db = GamsPandasDatabase()
  t = pd.Index([str(i) for i in range(2010, 2026)], name="t")
  par = pd.Series(1.4, index=t, name="par")
  db.add_parameter_from_series(par, add_missing_domains=True)
  assert all(db["par"] == 1.4)
  assert len(db["par"]) == 16


def test_multiply_with_different_sets():
  assert approximately_equal(
    sum(Gdx("test.gdx")["qBNP"] * Gdx("test.gdx")["qI"] * Gdx("test.gdx")["qI_s"]),
    50730260150
  )


def test_export_with_no_changes():
  Gdx("test.gdx").export("test_export.gdx", relative_path=True)
  assert round(os.stat("test.gdx").st_size, -5) == round(os.stat("test_export.gdx").st_size, -5)


def test_export_with_changes():
  gdx = Gdx("test.gdx")
  gdx["qY"] = gdx["qY"] * 2
  gdx.export("test_export.gdx", relative_path=True)

  old, new = Gdx("test.gdx"), Gdx("test_export.gdx")
  for i in old["qY"].index:
    approximately_equal(old["qY"][i] * 2, new["qY"][i])


def test_export_added_variable():
  gdx = Gdx("test.gdx")
  gdx.add_variable_dc(identifier="foo", domain_names=["a", "t"], explanatory_text="Variable added from Python.")
  gdx["foo"] = 42
  gdx.export("test_export.gdx", relative_path=True)
  assert all(approximately_equal(Gdx("test_export.gdx")["foo"], 42))


def test_detuple():
  assert GamsPandasDatabase.detuple("aaa") == "aaa"
  assert GamsPandasDatabase.detuple(("aaa",)) == "aaa"
  assert list(GamsPandasDatabase.detuple(("aaa", "bbb"))) == ["aaa", "bbb"]
  assert GamsPandasDatabase.detuple(1) == 1
  assert list(GamsPandasDatabase.detuple([1, 2])) == [1, 2]
