import plotly.offline as pyo
from plotly.subplots import make_subplots
from math import ceil


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
