import os

import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import dreamtools as dt

dt.REFERENCE_DATABASE = r = dt.Gdx("test.gdx")
s = dt.Gdx("test.gdx")

def test_to_dataframe():
  dt.time(2025, 2040)
  assert dt.to_dataframe(s.qBNP).shape == (16, 1)
  assert dt.to_dataframe(s.qY).shape == (16, 1)
  assert dt.to_dataframe(s.qY[s.s]).shape == (16, 9)
  assert dt.to_dataframe(s.qY[s.s], "q").shape == (16, 9)
  s.foo = s.qY.loc[s.sp]
  s.foo.name = "foo"
  assert dt.to_dataframe(s.foo).shape == (16, 8)
  dt.to_dataframe(s.foo, "q")
