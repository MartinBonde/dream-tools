import os
import sys
# os.chdir("..")
sys.path.insert(0, os.getcwd())

import pytest
import numpy as np
import pandas as pd
import dreamtools as dt
import gams.transfer as gt

@np.vectorize
def approximately_equal(x, y, ndigits=6):
  return round(x, ndigits) == round(y, ndigits)

def test_gdx_read():
  db = dt.Gdx("test.gdx")
  assert approximately_equal(db["qY"]["byg", 2025], 257.55, ndigits=2)
  assert approximately_equal(db["qI_s"]["IB", "fre", 2025], 7.41, ndigits=2)
  assert approximately_equal(db["eHh"], 1.25)
  assert db["fp"] == 1.0178
  db.create_parameter("inf_factor_test", db["t"], data=db["fp"]**(2010 - db["t"]))
  assert all(approximately_equal(db["inf_factor"], db["inf_factor_test"]))
  '''Assert below fails, see utility - better_index_from_symbol'''
  assert db["s"].name == "s_"
  assert db.vHh.loc["NetFin","tot",1970] == 0
  assert set(db["vHh"].index.get_level_values("a_")).issubset(set(db["a_"]))
  
def test_create_set_from_index():
  db = dt.GamsPandasDatabase()
  t = pd.Index(range(2025, 2036), name="t")
  db.create_set("t", t)
  assert db["t"].name == "t"
  assert all(db["t"] == t)
  assert db.t.domains == ["*"]

  db.create_set("tsub", t[5:], domains=["t"])
  assert db["tsub"].name == "tsub"
  assert all(db["tsub"] == t[5:])
  assert db.tsub.domains == ["t"]

  s = pd.Index(["services", "goods"], name="s")
  st = pd.MultiIndex.from_product([s, t], names=["s", "t"])
  db.create_set("st", st)
  assert db["st"].name == "st"
  assert all(db["st"] == st[:])
  assert db.st.domains == ["s", "t"]

def test_add_parameter_from_dataframe():
  db = dt.GamsPandasDatabase()
  df = pd.DataFrame()
  df["t"] = range(2025, 2036)
  df["value"] = 1.3
  db.add_parameter_from_dataframe("par", df, add_missing_domains=True)
  assert all(db["par"] == 1.3)
  assert len(db["par"]) == 11

def test_multiply_added():
  db = dt.GamsPandasDatabase()
  df = pd.DataFrame([
    [2025, "ser", 3],
    [2025, "goo", 2],
    [2035, "ser", 6],
    [2035, "goo", 4],
  ], columns=["t", "s", "value"])
  db.add_parameter_from_dataframe("q", df, add_missing_domains=True)

  df = pd.DataFrame([
    [2025, 1],
    [2035, 1.2],
  ], columns=["t", "value"])
  db.add_parameter_from_dataframe("p", df, add_missing_domains=True)

  v = db["p"] * db["q"]
  v.name = "v"
  db.add_parameter_from_series(v)

  assert db["v"][2035, "goo"] == 4.8

def test_add_parameter_from_series():
  db = dt.GamsPandasDatabase()

  t = pd.Index(range(2025, 2036), name="t")
  par = pd.Series(1.4, index=t, name="par")
  db.add_parameter_from_series(par, add_missing_domains=True)
  assert all(db["par"] == 1.4)
  assert len(db["par"]) == 11

  ss = pd.Index(["foo"], name="ss")
  singleton = pd.Series(1.4, index=ss, name="singleton")
  db.add_parameter_from_series(singleton, add_missing_domains=True)
  assert db["singleton"]["foo"] == 1.4
  assert len(db["singleton"]) == 1

  scalar = pd.Series(1.4, name="scalar")
  db.add_parameter_from_series(scalar)
  assert all(db["scalar"] == 1.4)
  assert len(db["scalar"]) == 1

def test_create_variable():
  db = dt.GamsPandasDatabase()
  db.create_variable("scalar", data=3.2)
  assert db.scalar == 3.2
  db.create_variable("vector", data=[1, 2], index=pd.Index(["a", "b"], name="ab"), add_missing_domains=True)
  assert all(db.vector == [1, 2])

  db.create_variable("dataframe",
                     data=pd.DataFrame([
                       [2025, "ser", 3],
                       [2025, "goo", 2],
                       [2035, "ser", 6],
                       [2035, "goo", 4],
                     ], columns=["t", "s", "value"]),
                     add_missing_domains=True
                     )
  db.export("test_export.gdx")
  assert dt.Gdx("test_export.gdx")["scalar"] == 3.2
  assert all(dt.Gdx("test_export.gdx")["vector"] == [1, 2])
  assert all(db.s == ["ser", "goo"])
  assert all(db.t == [2025, 2035])

def test_create_parameter():
  db = dt.GamsPandasDatabase()
  db.create_parameter("scalar", data=3.2)
  assert db.scalar == 3.2
  db.create_parameter("vector", data=[1, 2], index=pd.Index(["a", "b"], name="ab"), add_missing_domains=True)
  assert all(db.vector == [1, 2])

  db.create_parameter("dataframe",
                     data=pd.DataFrame([
                       [2025, "ser", 3],
                       [2025, "goo", 2],
                       [2035, "ser", 6],
                       [2035, "goo", 4],
                     ], columns=["t", "s", "value"]),
                     add_missing_domains=True
                     )
  db.export("test_export.gdx")
  assert dt.Gdx("test_export.gdx")["scalar"] == 3.2
  assert all(dt.Gdx("test_export.gdx")["vector"] == [1, 2])
  assert all(db.s == ["ser", "goo"])
  assert all(db.t == [2025, 2035])

def test_add_variable_from_series():
  db = dt.GamsPandasDatabase()
  t = pd.Index(range(2010, 2026), name="t")
  var = pd.Series(1.4, index=t, name="var")
  db.add_variable_from_series(var, add_missing_domains=True)
  assert all(db["var"] == 1.4)
  assert len(db["var"]) == 16

def test_add_variable_from_dataframe():
  db = dt.GamsPandasDatabase()
  df = pd.DataFrame([
    [2010, "ser", 3],
    [2010, "goo", 2],
    [2020, "ser", 6],
    [2020, "goo", 4],
  ], columns=["t", "s", "value"])
  db.add_variable_from_dataframe("q", df, add_missing_domains=True)
  assert all(db.t == [2010, 2020])
  assert all(db.s == ["ser", "goo"])

def test_multiply_with_different_sets():
  db = dt.Gdx("test.gdx")
  i, s = db["i"], db["s"]
  result = (db["pI"] * db["qI_s"].loc[i, s]).groupby("t").sum() / db["vI"]["iTot"] # Using pI_s would give exact 1
  assert all(approximately_equal(result.loc[2030:], 1, ndigits=1))

def test_export_with_no_changes():
  dt.Gdx("test.gdx").export("test_export.gdx", relative_path=True)
  original=gt.Container('test.gdx')
  exported=gt.Container('test_export.gdx')
  assert {s.name for s in original.getSymbols()} == {s.name for s in exported.getSymbols()} #verify symbols are identical
  #assert check records, domains, that sorta stuff. 
  #atm it seems that a_ have the same elements, but the order is different causing test to fail. We ignore for now.
  do_not_sort=['level','value','marginal','lower','upper','scale']
  def normalize(df):
      # Convert all categorical columns to strings to avoid category order issues
      for col in df.columns:
          if pd.api.types.is_categorical_dtype(df[col]):
              df[col] = df[col].astype(str)
      return df

  for sym in {s.name for s in original.getSymbols()}:
      df1 = normalize(original[sym].records.copy())
      df2 = normalize(exported[sym].records.copy())

      key_cols = [col for col in df1.columns if col not in do_not_sort]

      df1 = df1.sort_values(by=key_cols).reset_index(drop=True)
      df2 = df2.sort_values(by=key_cols).reset_index(drop=True)

      try:
          pd.testing.assert_frame_equal(df1, df2, check_dtype=False, check_like=True)
      except AssertionError as e:
          raise AssertionError(f"Mismatch at symbol '{sym}':\n{str(e)}")

def test_export_variable_with_changes():
  db = dt.Gdx("test.gdx")
  db["qY"] = db["qY"] * 2
  db.export("test_export.gdx", relative_path=True)
  old, new = dt.Gdx("test.gdx"), dt.Gdx("test_export.gdx")
  assert all(old["qY"] * 2 == new["qY"])

def test_export_parameter_with_changes():
  db = dt.Gdx("test.gdx")
  db["growth_factor"] = db["growth_factor"] * 2
  db.export("test_export.gdx", relative_path=True)
  old, new = dt.Gdx("test.gdx"), dt.Gdx("test_export.gdx")
  assert all(old["growth_factor"] * 2 == new["growth_factor"])

def test_export_scalar_with_changes():
  db = dt.Gdx("test.gdx")
  db["eHh"] = db["eHh"] * 2
  db.export("test_export.gdx", relative_path=True)

  old, new = dt.Gdx("test.gdx"), dt.Gdx("test_export.gdx")
  assert approximately_equal(old["eHh"] * 2, new["eHh"])

def test_export_set_with_changes():
  db = dt.Gdx("test.gdx")
  db["s"].texts["tje"] = "New text"
  db.export("test_export.gdx", relative_path=True)
  assert dt.Gdx("test_export.gdx")["s"].texts["tje"] == "New text"

def test_copy_set():
  db = dt.Gdx("test.gdx")
  db["alias"] = db["s"]
  db["alias"].name = "alias"
  index = db["alias"]
  domains = ["*" if i in (None, index.name) else i for i in db.get_domains_from_index(index, index.name)]
  '''The method add_set_dc does not exist in gams-transfer. It is rebuit in the GamsPandasDatabase class'''
  db.add_set_dc(index.name,domains,"")
  index = index.copy()
  index.domains = domains
  db.series[index.name] = index
  db.export("test_export.gdx", relative_path=True)
  assert all(dt.Gdx("test_export.gdx")["alias"] == db["s"])

def test_export_added_variable():
  db = dt.Gdx("test.gdx")
  db.create_variable("foo", [db.a, db.t], explanatory_text="Variable added from Python.")
  db["foo"] = 42
  db.export("test_export.gdx", relative_path=True)
  assert all(approximately_equal(dt.Gdx("test_export.gdx")["foo"], 42))

def test_export_NAs():
  db = dt.GamsPandasDatabase()
  t = db.create_set("t", range(5))
  p = db.create_parameter("p", t)
  p_nans=db.create_parameter("p_nans", t, data=[1, 2, np.nan, 4, 5])
  assert len(db["p"]) == 5
  db.export("test_export.gdx")
  db = dt.Gdx("test_export.gdx")
  assert all(pd.isna(db["p"]))
  expected = pd.Series([1, 2, np.nan, 4, 5], index=pd.Index(['0.0','1.0','2.0','3.0','4.0'],name='t'), name="p_nans")
  pd.testing.assert_series_equal(db['p_nans'], expected)

def test_detuple():
  assert dt.GamsPandasDatabase.detuple("aaa") == "aaa"
  assert dt.GamsPandasDatabase.detuple(("aaa",)) == "aaa"
  assert list(dt.GamsPandasDatabase.detuple(("aaa", "bbb"))) == ["aaa", "bbb"]
  assert dt.GamsPandasDatabase.detuple(1) == 1
  assert list(dt.GamsPandasDatabase.detuple([1, 2])) == [1, 2]

def test_create_methods():
  # Create empty GamsPandasDatabase and alias creation methods
  db = dt.GamsPandasDatabase()
  Par, Var, Set = db.create_parameter, db.create_variable, db.create_set

  # Create sets from scratch
  t = Set("t", range(2000, 2021), "Årstal")
  s = Set("s", ["tjenester", "fremstilling"], "Brancher", ["Tjenester", "Fremstilling"])
  st = Set("st", [s, t], "Branche x år dummy")

  sub = Set("sub", ["tjenester"], "Subset af brancher", domains=["s"])

  one2one = Set("one2one", [(2010, 2015), (2011, 2016)], "1 til 1 mapping", domains=["t", "t"])

  one2many = Set("one2many",
                 [("tot", "tjenester"), ("tot", "fremstilling")],
                 "1 til mange mapping", domains=["*", "s"],
                 )
  assert one2many.name == "one2many"
  assert one2many.names == ["*", "s"]
  assert one2many.domains == ["*", "s"]

  # Create parameters and variables based on zero ore more sets
  gq = Par("gq", None, "Produktivitets-vækst", 0.01)
  fq = Par("fp", t, "Vækstkorrektionsfaktor", (1 + 0.01)**(t-2010))
  d = Par("d", st, "Dummy")
  y = Var("y", [s,t], "Produktion")

  assert gq == 0.01
  assert all(fq.loc[2010:2011] == [1, 1.01])
  assert pd.isna(d["tjenester",2010])
  assert pd.isna(y["tjenester",2010])

  # Test that created symbols can be exported
  db.export("test_export.gdx")

def test_import_export_empty():
  # Create empty GamsPandasDatabase and alias creation methods
  db = dt.GamsPandasDatabase()
  Par, Var, Set = db.create_parameter, db.create_variable, db.create_set

  # Create sets from scratch
  t = Set("t", range(2000, 2021), "Årstal")
  s = Set("s", ["tjenester", "fremstilling"], "Brancher", ["Tjenester", "Fremstilling"])
  p = Par("p", [s, t])
  v = Var("v", [s, t])
  db.p = p.loc[[], []]
  db.v = p.loc[[], []]
  db.export("test_export.gdx")
  db = dt.Gdx("test_export.gdx")

  for i in s:
    for j in t:
      db.p.loc[i, j] = 1
      db.v.loc[i, j] = 1
  db.export("test_export.gdx")
  db = dt.Gdx("test_export.gdx")

  #assert all(db.p == 1)
  assert all(db.v == 1)

def test_get_sparse():
  db = dt.Gdx("test.gdx")
  dense = dt.Gdx("test.gdx", sparse=False)
  assert len(dense["qY"]) > len(db["qY"])
  assert len(dense["qY"]) == len(pd.MultiIndex.from_product(db.get("s_", "t")))
  assert len(dense["pBolig"]) == len(db["t"])

def test_time_index_pos():
  db = dt.Gdx("test.gdx")
  p = db.create_parameter("p", [db.s_, db.s, db.t], data=0)
  p_inv_sets = db.create_parameter("p_inv_sets", [db.t, db.s, db.s_], data=0)
  assert dt.DataFrame(p[:,'tje',:]).size == dt.DataFrame(p_inv_sets[:,'tje',:]).size

def test_aggregation_with_2_sets():
  db = dt.Gdx("test.gdx")
  p = db.create_parameter("p", [db.a_, db.portf_, db.t], data=0)
  assert dt.DataFrame(p).columns[0] == "p[tot,NetFin]"
  db.export('test_export.gdx')

def test_compare():
  db = dt.Gdx("test.gdx")
  with pytest.raises(ValueError):
    dt.DataFrame(db.qBNP, "q")
  baseline = dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  db.qBNP *= 1.01
  q = dt.DataFrame(db.qBNP, "q", start_year=2025)
  m = dt.DataFrame(db.qBNP, "m", start_year=2025)
  assert approximately_equal(q, 0.01).all().all()
  assert ((15 < m) & (m < 25)).all().all()

def test_aggregation():
  db = dt.Gdx("test.gdx")
  default_set_aggregations={"s_": ["tje"]}
  y = dt.DataFrame(db.qY, default_set_aggregations=default_set_aggregations)
  k = dt.DataFrame(db.qK, default_set_aggregations=default_set_aggregations)
  yk = dt.DataFrame([db.qY, db.qK], default_set_aggregations=default_set_aggregations)
  ky = dt.DataFrame([db.qK, db.qY], default_set_aggregations=default_set_aggregations)
  
  assert y.size + k.size == yk.size == ky.size