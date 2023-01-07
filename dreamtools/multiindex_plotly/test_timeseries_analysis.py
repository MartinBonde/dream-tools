import os
import sys
sys.path.insert(0, os.getcwd())
os.chdir("../..")

import pytest
import numpy as np
import pandas as pd
import dreamtools as dt

def test_to_dataframe_with_series():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  dt.time(2025, 2040)
  assert dt.to_dataframe(s.qBNP).shape == (16, 1)
  assert dt.to_dataframe(s.qY).shape == (16, 1)
  assert dt.to_dataframe(s.qY[s.s]).shape == (16, 9)
  assert dt.to_dataframe(s.qY[s.s], "q").shape == (16, 9)
  s.foo = s.qY.loc[s.sp]
  s.foo.name = "foo"
  assert dt.to_dataframe(s.foo).shape == (16, 8)
  with pytest.raises(KeyError):
    dt.to_dataframe(s.foo, "q")

def test_to_dataframe_with_database():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  dt.time(2025, 2040)
  assert dt.to_dataframe(s, function = lambda s: s.qBNP).shape == (16, 1)
  assert dt.to_dataframe(s, function = lambda s: s.qY).shape == (16, 1)
  assert dt.to_dataframe(s, function = lambda s: s.qY[s.s]).shape == (16, 9)
  assert dt.to_dataframe(s, "q", lambda s: s.qY[s.s]).shape == (16, 9)
  assert dt.to_dataframe(s, "q", lambda s: s.pY[s.s] * s.qY[s.s]).shape == (16, 9)
  
def test_to_dataframe_with_multiple_baselines():
  b1 = dt.Gdx("test.gdx")
  b2 = dt.Gdx("test.gdx")
  s1 = dt.Gdx("test.gdx")
  s2 = dt.Gdx("test.gdx")
  s1.reference_database = b1
  s2.reference_database = b2
  b1.qY *= 2
  b2.qY /= 2
  dt.time(2025, 2040)
  assert dt.to_dataframe([s1, s2], "q", lambda s: s.pY[s.s] * s.qY[s.s]).shape == (16, 9*2)
  q = dt.to_dataframe([s1, s2], "q", lambda s: s.pY['tje'] * s.qY['tje'], ["s1", "s2"])
  assert all(q["s1"] == -0.5)
  assert all(q["s2"] == 1)

  s1 = dt.Gdx("test.gdx")
  s2 = dt.Gdx("test.gdx")
  q = dt.to_dataframe([s1, s2], "q", lambda s: s.pY['tje'] * s.qY['tje'], ["s1", "s2"], baselines=[b1, b2])
  assert all(q["s1"] == -0.5)
  assert all(q["s2"] == 1)
