from math import ceil

from plotly import graph_objects as go, offline as pyo
from plotly.subplots import make_subplots

import pandas as pd

import dreamtools as dt

def small_figure(fig):
    "Adjust figure with settings suitable for side by side use in the Office suite"
    legend_height = 200 / dt.PLOT_SCALE + len(fig.data) * 64 / dt.PLOT_SCALE
    width, height = dt.SMALL_PLOT_WIDTH / dt.PLOT_SCALE, dt.PLOT_HEIGHT / dt.PLOT_SCALE
    fig = go.Figure(fig).update_layout(
        width = width,
        height = height + legend_height,
        legend_y = - 200 / dt.PLOT_HEIGHT,
        legend_yanchor = "top",
        legend_orientation = "v",
        margin = {"t": 20, "b": legend_height, "l": 60, "r": 10},

        # Title is used as y-axis label
        yaxis_title_text = "",
        title_text = fig.layout.yaxis.title.text,
        title_xanchor = "left", title_x = 0,
        title_yanchor = "top", title_y = 1,
        title_pad_l = 7, title_pad_t = 7,
        title_font_size = 14,
    )
    if  max(len(i.name) for i in fig.data) > 30:  # Long legend entries cause the plot area size to change when centered
        fig.update_layout(legend_x = 0, legend_xanchor = "left")
    return fig

def large_figure(fig):
    "Adjust figure with settings suitable for use in the Office suite"
    trace_count = len(fig.data)
    col_count = 2
    row_count = trace_count // col_count
    for i, trace in enumerate(fig.data):
        trace.legendgroup = i // (trace_count / col_count)
    legend_height = 200 / dt.PLOT_SCALE + row_count * 64 / dt.PLOT_SCALE
    width, height = dt.LARGE_PLOT_WIDTH / dt.PLOT_SCALE, dt.PLOT_HEIGHT / dt.PLOT_SCALE
    fig = go.Figure(fig).update_layout(
        width = width,
        height = height + legend_height,
        legend_y = - 200 / dt.PLOT_HEIGHT,
        legend_yanchor = "top",
        margin = {"t": 20, "b": legend_height, "l": 60, "r": 10},

        # Title is used as y-axis label
        yaxis_title_text="",
        title_text=fig.layout.yaxis.title.text,
        title_xanchor="left", title_x=0,
        title_yanchor="top", title_y=1,
        title_pad_l=7, title_pad_t=7,
        title_font_size=14,
    )
    return fig

def figures_to_html(figs, filename="figures.html"):
  """Write an iter of plotly figures to an html file."""
  file = open(filename, 'w')
  file.write("<html><head></head><body>" + "\n")

  for i, fig in enumerate(figs):
    inner_html = pyo.plot(
      fig, include_plotlyjs=(i==0), output_type='div'
    )
    file.write(inner_html)

  file.write("</body></html>" + "\n")

def subplot(figures, cols=2, **kwargs):
  """Create subplot from iter of figures."""
  rows = ceil(len(figures) / cols)
  fig = make_subplots(rows=rows,
                      cols=cols,
                      subplot_titles=[f["layout"]["title"]["text"] for f in figures],
                      **kwargs
                      )
  for i, f in enumerate(figures):
    for trace in f["data"]:
      fig.add_trace(trace, row=ceil((i+1)/cols), col=1 + i % cols)
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