import plotly.graph_objects as go
import dreamtools as dt
import pandas as pd

# def age_plot_3d(series, start_year=None, end_year=None, start_age=None, end_age=None, title="", ztitle="", **kwargs):
#   age_figure_3d(series, start_year, end_year, start_age, end_age, title, ztitle).show()
#
# def age_plot_2d(iter_series, operator=None, years=None, start_age=None, end_age=None, title="", reference_database=None):
#   age_figure_2d(iter_series, operator, years, start_age, end_age, title, reference_database).show()

def age_figure_3d(series,
                  start_year=None,
                  end_year=None,
                  start_age=None,
                  end_age=None,
                  title="",
                  ztitle="",
                  showscale=False,
                  **kwargs):
  if start_year is None:
    start_year = max(dt.START_YEAR, min(series.index.levels[-1]))
  if end_year is None:
    end_year = min(dt.END_YEAR, max(series.index.levels[-1]))
  if start_age is None:
    start_age = dt.START_AGE
  if end_age is None:
    end_age = dt.END_AGE

  age = list(range(start_age, end_age + 1))
  time = list(range(start_year, end_year + 1))
  value = series.loc[age, time].unstack().values
  surface = go.Surface(x=time, y=age, z=value, showscale=showscale, **kwargs)
  return go.Figure(
    surface,
    layout={
      "scene": {
        "xaxis": {"title": dt.TIME_AXIS_TITLE, "autorange": "reversed"},
        "yaxis": {"title": dt.AGE_AXIS_TITLE, "autorange": "reversed"},
        "zaxis": {"title": ztitle},
      },
      "title": {"text": title, 'x': 0.5, "y": 0.925}
    },
  )

def age_figure_2d(iter_series,
                  operator=None,
                  years=None,
                  start_age=None, end_age=None,
                  reference_database=None,
                  **kwargs
                  ):
  if isinstance(iter_series, pd.Series):
    iter_series = [iter_series]
  if years is None:
    years = list(range(dt.START_YEAR, dt.END_YEAR+1))
    if max(len(s.loc[:,years].index.unique(level=-1)) for s in iter_series) > 5:
      years = list(range(dt.START_YEAR, dt.END_YEAR + 1, 5))
  if start_age is None:
    start_age = dt.START_AGE
  if end_age is None:
    end_age = dt.END_AGE
  iter_series = [series.sort_index().loc[start_age:end_age, years] for series in iter_series]
  if operator:
    if reference_database is None:
      reference_database = dt.get_reference_database()
    refs = [reference_database[series.name].sort_index().loc[series.index] for series in iter_series]
    iter_series = dt.compare(iter_series, refs, operator)
  df = pd.DataFrame()
  for series in iter_series:
    series_df = series.unstack()
    for col in series_df:
      df[f"{series.name}[{col}]"] = series_df[col]
  return df.plot(**kwargs)