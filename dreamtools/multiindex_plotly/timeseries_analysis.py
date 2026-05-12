import dreamtools as dt
import numpy as np
import pandas as pd
import textwrap
from inspect import signature
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def get_reference_database(s=None):
  """Get baseline database associated with a GamsPandasDatabase. Defaults to dt.REFERENCE_DATABASE."""
  if isinstance(s, dt.GamsPandasDatabase) and s.reference_database is not None:
    return s.reference_database
  else:
    return dt.REFERENCE_DATABASE

def time(start, end=None):
  """Set global time settings."""
  if end is None:
    end = start
  dt.START_YEAR = start
  dt.END_YEAR = end

def years():
  """Return list of years in current time settings."""
  return list(range(dt.START_YEAR, dt.END_YEAR+1))

def foo(series):
  """For each level in series, index by ["tot"] if it exists, otherwise index by first element."""
  return series.loc[series.index.get_level_values(series.index.names[0]).isin(["tot"])]

def aggregate_series(series, default_set_aggregations=None):
  """Aggregate series according to default_set_aggregations."""
  if default_set_aggregations is None:
    default_set_aggregations = dt.DEFAULT_SET_AGGREGATIONS

  aggregated = series.copy()
  levels = aggregated.index.levels if aggregated.index.nlevels > 1 else [aggregated.index]
  for level in levels:
    aggregated_level = default_set_aggregations.get(level.name, list(level))
    mask = aggregated.index.get_level_values(level.name).isin(aggregated_level)
    if len(aggregated[mask]) > 0:
      aggregated = aggregated.loc[mask]

  return aggregated

def map_with_baseline(function, data, baselines):
  """Map function to data, passing baselines if function takes two arguments."""
  if len(signature(function).parameters) == 2:
    return list(map(function, data, baselines))
  else:
    return list(map(function, data))

@pd.api.extensions.register_dataframe_accessor("dt")
class _DataFrame(pd.DataFrame):
  """Pandas DataFrame with additional attributes for plotly layout."""

  _internal_names = pd.DataFrame._internal_names + ["layout"]
  _internal_names_set = set(_internal_names)

  @property
  def _constructor(self):
      return _DataFrame

  def plot(
    self,
    operator=None,
    layout=None,
    xline=None,
    vertical_legend=False,
    horizontal_yaxis_title=False,
    figure_size="document_large",
    legend_label_width="auto",
    colored_legend=True,
    alternating_dash=None,
    showlegend=None,
    **kwargs
  ):
    """Plot DataFrame using plotly."""
    layout = dict(layout or {})
    if showlegend is not None:
      layout["showlegend"] = showlegend

    fig = pd.DataFrame.plot(self, **kwargs)()

    fig.update_layout(**self.layout)

    if vertical_legend:
      fig = dt.vertical_legend(fig)

    if xline is not None:
      fig = add_xline(fig, xline)

    fig.update_layout(**dt.figure_layouts[figure_size])
    axis_title_font_sizes = {
      "xaxis": fig.layout.xaxis.title.font.size,
      "yaxis": fig.layout.yaxis.title.font.size,
    }

    fig.update_layout(**layout)
    if axis_title_font_sizes["xaxis"] is not None and fig.layout.xaxis.title.font.size is None:
      fig.update_layout(xaxis_title_font_size=axis_title_font_sizes["xaxis"])
    if axis_title_font_sizes["yaxis"] is not None and fig.layout.yaxis.title.font.size is None:
      fig.update_layout(yaxis_title_font_size=axis_title_font_sizes["yaxis"])
    if fig.layout.showlegend is None:
      fig.update_layout(showlegend=True)

    if horizontal_yaxis_title:
      fig = dt.horizontal_yaxis_title(fig)

    if operator == "s" and alternating_dash is None:
      alternating_dash = "dot"

    if alternating_dash is not None:
      fig = dt.alternating_dash(fig, dash=alternating_dash, line_width=3)

    if fig.layout.showlegend and legend_label_width:
      fig = dt.wrap_legend_labels(fig, width=legend_label_width)

    fig = compact_xaxis_title(fig)

    if fig.layout.showlegend:
      if colored_legend:
        fig = dt.colored_text_legend(fig)
      else:
        fig = dt.reserve_legend_space(fig)

    return fig

def alternating_dash(fig, dash="dot", line_width=None):
  """
  Update traces in plotly figure to have alternating dash styles and reusing colors.
  """
  line_colors = [trace.line.color for trace in fig.data]
  for i, trace in enumerate(fig.data):
    trace.update(
      line_dash = "solid" if i % 2 == 0 else dash,
      line_color = line_colors[i//2],
      line_width = line_width,
    )
  return fig

def horizontal_yaxis_title(fig, text=None):
  """
  Update plotly figure to make the y-axis horizontal using annotations
  """
  if text is None:
    text = fig.layout.yaxis.title.text
  return fig.update_layout(
    yaxis_title_text = "",
    annotations = [
      dict(
        x = 0, xshift = - 0.8 * fig.layout.margin.l, xref = "paper",
        y = 1, yshift = 0.8 * fig.layout.margin.t, yref = "paper",     
        text = text,
        showarrow = False,
      )
    ]
  )

def vertical_legend(fig, col_count=2):
  """
  Update plotly figure by splitting legend into <col_count> columns.
  """
  trace_count = len(fig.data)
  for i, trace in enumerate(fig.data):
    trace.legendgroup = i // (trace_count / col_count)
  return fig

def wrap_legend_labels(fig, width="auto"):
  """
  Wrap long legend labels so horizontal legends do not overlap.
  """
  if width == "auto":
    font_size = fig.layout.legend.font.size or fig.layout.font.size or 10
    width = legend_label_character_count(fig, font_size)

  for trace in fig.data:
    if trace.name:
      trace.name = "<br>".join(textwrap.wrap(
        trace.name,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
      ))
  return fig

def reserve_legend_space(fig, columns=None):
  """
  Increase figure height and bottom margin so the chart area stays fixed when the legend grows.
  """
  traces = [trace for trace in fig.data if trace.showlegend is not False and trace.name]
  if not traces:
    return fig

  font_size = fig.layout.legend.font.size or fig.layout.font.size or 10
  columns = columns or legend_column_count(fig, traces, font_size)
  line_height = 1.25 * font_size
  row_gap = 0.6 * font_size
  legend_rows = [
    traces[i:i + columns]
    for i in range(0, len(traces), columns)
  ]
  legend_height = sum(
    max(trace.name.count("<br>") + 1 for trace in row) * line_height + row_gap
    for row in legend_rows
  )

  margin = fig.layout.margin
  old_margin_b = margin.b or 0
  new_margin_b = old_margin_b + legend_height + _extra_gap_below_xaxis(fig)
  return fig.update_layout(
    height=(fig.layout.height or 0) + new_margin_b - old_margin_b,
    margin_b=new_margin_b,
  )

def legend_label_lines(trace):
  return trace.name.split("<br>")

def legend_label_width(trace, font_size):
  return max(map(len, legend_label_lines(trace))) * font_size * 0.55

def legend_entry_width_px(fig, trace, columns, font_size, needs_line_samples):
  """Horizontal space for one legend entry; matches legend_entry_layout (inset + segment + gap + label)."""
  w = legend_label_width(trace, font_size)
  if not needs_line_samples:
    return w
  pw = legend_width(fig)
  gap_px = 0.5 * font_size
  if not pw:
    return w + 2.4 * font_size + 2 * gap_px
  segment_px = min(2.4 * font_size, pw * (0.5 / columns))
  return w + gap_px + segment_px + gap_px

def legend_label_character_count(fig, font_size):
  traces = [trace for trace in fig.data if trace.showlegend is not False and trace.name]
  available_width = legend_width(fig)
  if not available_width:
    return 48
  sample_width = 2.9 * font_size if legend_needs_line_samples(traces) else 0
  return max(1, int((available_width - sample_width) / (font_size * 0.55)))

def _extra_gap_below_xaxis(fig):
  """Extra vertical gap (px) between x-axis tick labels and the colored legend; covers x-axis title."""
  title = fig.layout.xaxis.title
  text = title.text if title else None
  if not text or not str(text).strip():
    return 0
  fs = (title.font.size if title.font else None) or fig.layout.font.size or 10
  return 1.35 * fs

def compact_xaxis_title(fig, standoff_px=4):
  """Pull x-axis title closer to tick labels (Plotly default ~15px); frees space above the colored legend."""
  title = fig.layout.xaxis.title
  if not title or not title.text or not str(title.text).strip():
    return fig
  return fig.update_layout(xaxis=dict(title=dict(standoff=standoff_px)))

def legend_width(fig):
  margin = fig.layout.margin
  return (fig.layout.width or 0) - (margin.l or 0) - (margin.r or 0)

def legend_height(fig):
  margin = fig.layout.margin
  return (fig.layout.height or 0) - (margin.t or 0) - (margin.b or 0)

def legend_column_count(fig, traces, font_size):
  """Max columns such that each entry fits its column band (width / columns); not sum(widths) on one row."""
  available_width = legend_width(fig)
  if not available_width:
    return 1

  needs_line_samples = legend_needs_line_samples(traces)
  for columns in range(len(traces), 1, -1):
    col_w = available_width / columns
    if all(
      legend_entry_width_px(fig, trace, columns, font_size, needs_line_samples) <= col_w
      for trace in traces
    ):
      return columns
  return 1

def trace_dash(trace):
  return trace.line.dash or "solid"

def legend_needs_line_samples(traces):
  return any(trace_dash(trace) != "solid" for trace in traces)

def legend_sample_shape(trace, x0, x1, y, color="black", line_width=2):
  return dict(
    type="line",
    x0=x0,
    x1=x1,
    xref="paper",
    y0=y,
    y1=y,
    yref="paper",
    line=dict(
      color=color,
      dash=trace_dash(trace),
      width=trace.line.width or line_width,
    ),
  )

def legend_entry_layout(fig, trace, columns, column_number, yshift, font_size):
  plot_width = legend_width(fig)
  plot_height = legend_height(fig)
  if not plot_width or not plot_height:
    return (column_number + 0.5) / columns, "center", None

  sample_width = min(2.4 * font_size / plot_width, 0.5 / columns)
  label_gap = 0.5 * font_size / plot_width
  col_left = column_number / columns
  inset = label_gap
  x0 = col_left + inset
  x1 = x0 + sample_width
  y = (yshift - font_size) / plot_height
  shape = legend_sample_shape(trace, x0, x1, y)
  return x1 + label_gap, "left", shape

def legend_label_anchor_in_column(fig, columns, column_number, font_size):
  """Left-align labels within each column when columns > 1; single column stays centered."""
  if columns <= 1:
    return (column_number + 0.5) / columns, "center"
  plot_width = legend_width(fig)
  if not plot_width:
    return (column_number + 0.5) / columns, "center"
  inset = 0.5 * font_size / plot_width
  return column_number / columns + inset, "left"

def trace_color(fig, trace, i):
  color = getattr(trace.line, "color", None) or getattr(trace.marker, "color", None)
  if color is not None:
    return color
  colorway = fig.layout.colorway or fig.layout.template.layout.colorway or ()
  return colorway[i % len(colorway)] if colorway else "black"

def colored_text_legend(fig, columns=None):
  """
  Replace the native legend with colored text labels below the chart (single column centered; multiple columns left-aligned per column).
  """
  traces = [trace for trace in fig.data if trace.showlegend is not False and trace.name]
  if not traces:
    return fig

  font_size = fig.layout.legend.font.size or fig.layout.font.size or 10
  line_height = 1.25 * font_size
  row_gap = 0.6 * font_size
  axis_gap = 2.2 * font_size + _extra_gap_below_xaxis(fig)
  columns = columns or legend_column_count(fig, traces, font_size)
  legend_rows = [traces[i:i + columns] for i in range(0, len(traces), columns)]
  yshift = -(axis_gap + row_gap)
  annotations = list(fig.layout.annotations or [])
  shapes = list(fig.layout.shapes or [])
  show_line_samples = legend_needs_line_samples(traces)

  for row_number, row in enumerate(legend_rows):
    row_height = max(len(legend_label_lines(trace)) for trace in row) * line_height
    for column_number, trace in enumerate(row):
      i = row_number * columns + column_number
      shape = None
      if show_line_samples:
        x, xanchor, shape = legend_entry_layout(fig, trace, columns, column_number, yshift, font_size)
        shape["line"]["color"] = trace_color(fig, trace, i)
        shapes.append(shape)
      else:
        x, xanchor = legend_label_anchor_in_column(fig, columns, column_number, font_size)
      annotations.append(dict(
        x=x,
        xref="paper",
        xanchor=xanchor,
        y=0,
        yref="paper",
        yanchor="top",
        yshift=yshift,
        text=trace.name,
        font=dict(size=font_size, color=trace_color(fig, trace, i)),
        showarrow=False,
      ))
      trace.showlegend = False
    yshift -= row_height + row_gap

  margin = fig.layout.margin
  old_margin_b = margin.b or 0
  legend_height = -yshift
  new_margin_b = old_margin_b + legend_height
  return fig.update_layout(
    showlegend=False,
    annotations=annotations,
    shapes=shapes,
    height=(fig.layout.height or 0) + new_margin_b - old_margin_b,
    margin_b=new_margin_b,
  )

def add_xline(fig, x):
  "Add a vertical line to a plotly figure at x"
  return fig.update_layout(shapes=[dict(
    type='line',
    yref='paper', y0=0, y1=1,
    xref='x', x0=x, x1=x,
    line=dict(
      dash="dash",
    ),
    opacity=0.3,
  )])

def DataFrame(
  data,
  operator=None,
  function=None,
  names=None,
  start_year=None,
  end_year=None,
  baselines=None,
  default_set_aggregations=None,
  functions=None,
):
  if isinstance(data, pd.Series) or isinstance(data, dt.GamsPandasDatabase):
    data = [data]

  if function is not None:
    functions = [function]

  if functions is None:
    if isinstance(data[0], dt.GamsPandasDatabase):
      raise ValueError("Must specify function when passing GamsPandasDatabase.")
    functions = [lambda x: x]

  if baselines is None:
    baselines = [get_reference_database(s) for s in data]

  results = [map_with_baseline(function, data, baselines) for function in functions]

  if operator:
    if None in baselines:
      raise ValueError("Cannot compare with baseline as no baseline has been specified and no global reference database has been set.")
    if isinstance(data[0], pd.Series):
      baseline_series = map(get_baseline_series, data, baselines)
      baseline_results = [map_with_baseline(function, baseline_series, baselines) for function in functions]
    else:
      baseline_results = [map_with_baseline(function, baselines, baselines) for function in functions]
    results = [compare(a, b, operator) for a, b in zip(results, baseline_results)]

  aggregated = [aggregate_series(s, default_set_aggregations) for f in results for s in f]

  df = merge_multiseries(aggregated)
  if start_year is None:
    start_year = dt.START_YEAR
  if end_year is None:
    end_year = dt.END_YEAR
  df = df.loc[start_year:end_year]

  if names:
    df.columns = names

  df = _DataFrame(df)

  # Set default layout for plotly which depends on the operator
  df.layout = {
    "yaxis_title_text": dt.yaxis_title_from_operator(operator),
    "xaxis_title_text": dt.time_axis_title(),
    "legend_title_text": "",
  }

  return df

def get_baseline_series(x, b):
  """Lookup the name of series x in the reference database b and return the series from b with the same index as x"""
  if x.name not in b:
    raise KeyError(f"'{x.name}' was not found in the reference database.")

  y = b[x.name]
  if (
    y.index.nlevels != x.index.nlevels
    or y.index.names != x.index.names
  ):
    raise KeyError(f"The dimension of '{x.name}' is different in the reference database. If indexing a single element write [['element']] rather than ['element'] to prevent the series dimension being reduced.")

  return (x - x + y)[x.index] # Adding and subtracting x is a trick to keep the index

def compare(iter_series, refs, operator):
  """
  Applies an operator to each pair in zip(iter_series, refs)
  """
  if operator in ["q"]:
    return [s / b - 1 for s, b in zip(iter_series, refs)]
  elif operator in ["pq"]:
    return [(s / b - 1)*100 for s, b in zip(iter_series, refs)]
  elif operator in ["m"]:
    return [s - b for s, b in zip(iter_series, refs)]
  elif operator in ["pm"]:
    return [(s - b)*100 for s, b in zip(iter_series, refs)]
  elif operator in ["s"]:
    for b in refs:
      b.name = "baseline." + b.name
    return [i for pair in zip(iter_series, refs) for i in pair]
  elif operator in ["p"]:
    return [s / lag(s) * 100 - 100 for s in iter_series]
  elif operator in ["d"]:
    return [s - lag(s) for s in iter_series]
  elif operator in ["i"]:
    return [index(s) for s in iter_series]
  elif operator in ["log"]:
    return [np.log(s) for s in iter_series]
  elif operator in ["rlog"]:
    return [np.log(s) for s in refs]
  elif operator in ["dlog"]:
    return [np.log(s) - np.log(lag(s)) for s in iter_series]
  else:
    raise ValueError(f"{operator} is not a valid operator.")

def index(series, level=-1, element=None):
  """Return series divided by the value with the index element. E.g. set a series to 1 in a base year."""
  if element is None:
    element = dt.START_YEAR
  if len(series.index.names) > 1:
    return series / series.xs(element, level=level)
  else:
    return series / series[element]

def lag(series, periods=1, lag_axis_index=-1):
  if len(series.index.names) > 1:
    return series.groupby(level=series.index.names[:lag_axis_index:]).shift(periods)
  else:
    return series.shift(periods)

def unstack_multiseries(series, keep_axis_index=-1):
  """
  Return a DataFrame from a Series.
  All levels of the Series are unstacked and concatenated as column names, except <keep_axis_index>.
  """
  if isinstance(series.index, pd.MultiIndex):
    series = series.copy()
    keep_axis_name = series.index.names[keep_axis_index]
    if series.name is None:
      series.name = ""
    series.index = pd.MultiIndex.from_tuples(flatten_keys(series.name, keys, keep_axis_index) for keys in series.index)
    df = series.unstack(0)[series.index.get_level_values(0).unique()]
    df.index.name = keep_axis_name
  else:
    df = pd.DataFrame(series)
  return df

def flatten_keys(name, keys, keep_axis_index):
  keys_str = ','.join(map(str, [keys[i] for i, _ in enumerate(keys) if i != keep_axis_index]))
  flat_name = f"{name}[{keys_str}]"
  return flat_name, keys[keep_axis_index]

def merge_multiseries(series, keep_axis_indices=None):
  """
  Return a DataFrame from any number of Series, with all levels except <keep_axis_index> concatenated as column names.
  """
  output = pd.DataFrame()
  if keep_axis_indices is None:
    keep_axis_indices = [get_keep_axis_index(s) for s in series]
  for s, keep_axis_index in zip(series, keep_axis_indices):
    df = unstack_multiseries(s, keep_axis_index)
    for c in df.columns:
      new_name = c if (c != "0") else ""
      iter = 0
      while new_name in output:
        iter += 1
        new_name = f"{c}{iter}"
      output[new_name] = df[c]

  return output


def get_keep_axis_index(series):
  """
  Return the index of the axis to keep when unstacking a multi-indexed series.
  """
  try:
    return series.index.names.index(dt.X_AXIS_NAME)
  except ValueError:
    return dt.X_AXIS_INDEX # Default if no level named dt.X_AXIS_NAME is found
