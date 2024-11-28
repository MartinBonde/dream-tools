import os
import sys
# os.chdir("../..")
sys.path.insert(0, os.getcwd())

import pytest
import numpy as np
import pandas as pd
import dreamtools as dt

def test_DataFrame_with_series():
  dt.REFERENCE_DATABASE = dt.Gdx("../test.gdx")
  s = dt.Gdx("../test.gdx")
  dt.time(2025, 2040)
  assert dt.DataFrame(s.qBNP).shape == (16, 1)
  assert dt.DataFrame(s.qY).shape == (16, 1)
  assert dt.DataFrame(s.qY[s.s]).shape == (16, 9)
  assert dt.DataFrame(s.qY[s.s], "q").shape == (16, 9)
  assert dt.DataFrame(s.qY[s.s], "s").shape == (16, 9*2)
  s.foo = s.qY.loc[s.sp]
  s.foo.name = "foo"
  assert dt.DataFrame(s.foo).shape == (16, 8)
  with pytest.raises(KeyError):
    dt.DataFrame(s.foo, "q")

def test_DataFrame_with_database():
  dt.REFERENCE_DATABASE = dt.Gdx("../test.gdx")
  s = dt.Gdx("../test.gdx")
  dt.time(2025, 2040)
  assert dt.DataFrame(s, function = lambda s: s.qBNP).shape == (16, 1)
  assert dt.DataFrame(s, function = lambda s: s.qY).shape == (16, 1)
  assert dt.DataFrame(s, function = lambda s: s.qY[s.s]).shape == (16, 9)
  assert dt.DataFrame(s, "q", lambda s: s.qY[s.s]).shape == (16, 9)
  assert dt.DataFrame(s, "q", lambda s: s.pY[s.s] * s.qY[s.s]).shape == (16, 9)
  
def test_DataFrame_with_multiple_baselines():
  b1 = dt.Gdx("../test.gdx")
  b2 = dt.Gdx("../test.gdx")
  s1 = dt.Gdx("../test.gdx")
  s2 = dt.Gdx("../test.gdx")
  s1.reference_database = b1
  s2.reference_database = b2
  b1.qY *= 2
  b2.qY /= 2
  dt.time(2025, 2040)
  assert dt.DataFrame([s1, s2], "q", lambda s: s.pY[s.s] * s.qY[s.s]).shape == (16, 9*2)
  q = dt.DataFrame([s1, s2], "q", lambda s: s.pY['tje'] * s.qY['tje'], ["s1", "s2"])
  assert all(q["s1"] == -0.5)
  assert all(q["s2"] == 1)

  s1 = dt.Gdx("../test.gdx")
  s2 = dt.Gdx("../test.gdx")
  q = dt.DataFrame([s1, s2], "q", lambda s: s.pY['tje'] * s.qY['tje'], ["s1", "s2"], baselines=[b1, b2])
  assert all(q["s1"] == -0.5)
  assert all(q["s2"] == 1)

def test_functions():
  dt.REFERENCE_DATABASE = dt.Gdx("../test.gdx")
  db = dt.Gdx("../test.gdx")
  df1 = dt.DataFrame(db, functions=[lambda s: s.qBNP, lambda s: s.pBNP])
  df2 = dt.DataFrame([db.qBNP, db.pBNP])
  assert df1.equals(df2)

  df1 = dt.DataFrame(db, "pq", functions=[lambda s: s.qBNP, lambda s: s.pBNP])
  df2 = dt.DataFrame([db.qBNP, db.pBNP], "pq")
  assert df1.equals(df2)