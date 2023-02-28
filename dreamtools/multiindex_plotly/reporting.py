from math import ceil
from time import sleep
import inspect

from plotly import offline as pyo
from plotly.subplots import make_subplots

import pandas as pd

import dreamtools as dt

def plot(*args, **kwargs):
  """Shorthand for DataFrame(...).plot(...)"""
  dataframe_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(dt.DataFrame).parameters}
  plot_kwargs = {k: v for k, v in kwargs.items() if k not in dataframe_kwargs}
  dataframe = dt.DataFrame(*args, **dataframe_kwargs)
  return dataframe.plot(**plot_kwargs)

def prt(
  iter_series,
  operator=None,
  function=None,
  names=None,
  start_year=None,
  end_year=None,
  reference_database=None,
  default_set_aggregations=None,
  dec=6,
  max_rows=100,
  max_columns=20,
):
  """Print a table of a series or list of series."""
  df = dt.DataFrame(iter_series, operator, function, names, start_year, end_year, reference_database, default_set_aggregations)
  df.style.set_properties(**{"text-align": "right", "precision": dec})
  with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):  # more options can be specified also
    display(df)

def write_image(fig, file_name, scale=3):
  fig.write_image(file_name, scale=scale)
  if file_name.endswith(".png"):
    from PIL import Image
    with Image.open(file_name) as img:
      sleep(0.01)
      img.save(file_name, dpi=(96 * scale, 96 * scale))

def figures_to_html(figs, filename="figures.html", encoding="utf-8"):
  """Write an iter of plotly figures to an html file."""
  with open(filename, 'w', encoding=encoding) as file:
    file.write("<html><head></head><body>" + "\n")

    for i, fig in enumerate(figs):
      inner_html = pyo.plot(
        fig, include_plotlyjs=(i==0), output_type='div'
      )
      file.write(inner_html)

    file.write("</body></html>" + "\n")

def subplot(figures, n_columns=2, n_legend_columns=2, **kwargs):
  """Create subplot from iter of figures."""
  rows = ceil(len(figures) / n_columns)
  fig = make_subplots(
    rows=rows,
    cols=n_columns,
    subplot_titles=[f["layout"]["title"]["text"] for f in figures],
    **kwargs
  )
  for i, f in enumerate(figures):
    for trace in f["data"]:
      trace.showlegend = i==1
      fig.add_trace(trace, row=ceil((i+1)/n_columns), col=1 + i % n_columns)
  trace_count = len(figures[0].data)
  for i, trace in enumerate(fig.data):
    trace.legendgroup = i // (trace_count / n_legend_columns)
  return fig

def html_table(df, precision=2, header=True, header_background_color="#14AFA6", head_text_color="#ffffff"):
  """Return string with a formated HTML version of a Pandas Dataframe"""
  df = df.copy()
  df.index.name, df.columns.name = None, None

  styles = [
    # table properties
    dict(selector=" ",
         props=[
           ("margin", "0"),
           ("font-family", '"Hind", "Helvetica", "Arial", sans-serif'),
           ("border-collapse", "collapse"),
           # ("border", "2px solid #0F837D"),
           ("text-align", "right"),
         ]),

    # header color
    dict(selector="thead",
         props=[
           ("background-color", header_background_color),
           ("color", head_text_color),
         ]),

    # background shading
    dict(selector="tbody",
         props=[("background-color", "#E6E6E8")]),
    # dict(selector="tbody tr:nth-child(even)",
    #         props=[("background-color", "#fff")]),
    # dict(selector="tbody tr:nth-child(odd)",
    #         props=[("background-color", "#eee")]),

    # body cell properties
    dict(selector="td",
         props=[
           ("font-size", "120%"),
           ("padding", ".5em"),
           ("border", "1px solid #ffffff"),
         ]),

    # header cell properties
    dict(selector="th",
         props=[
           ("font-size", "120%"),
           ("text-align", "right"),
           ("padding", ".5em"),
           ("border", "1px solid #ffffff"),
         ]),

    # caption placement
    dict(selector="caption",
         props=[("caption-side", "bottom")]),
  ]
  if not header:
    styles.append({"selector": "thead", "props": [("font-size", "0pt")]})

  with pd.option_context("precision", precision):
    return "<meta charset='UTF-8'>" + df.style.set_table_styles(styles).render()