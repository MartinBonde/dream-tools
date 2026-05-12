import os
import re
import sys
# os.chdir("../..")
sys.path.insert(0, os.getcwd())

import pytest
import numpy as np
import pandas as pd
import dreamtools as dt
from PIL import Image
import xml.etree.ElementTree as ET

TEMPLATE_PDF_DIRS = [
  "I:/Skabeloner/Lyx/figures",
  "I:/Skabeloner/Latex/figures",
]

def test_language_settings():
  dt.set_language("en")
  assert dt.LANGUAGE == "en"
  assert dt.age_axis_title() == "age"
  assert dt.time_axis_title() == ""
  assert dt.yaxis_title_from_operator("m") == "Difference from baseline"

  dt.set_language("da")
  assert dt.age_axis_title() == "Alder"
  assert dt.time_axis_title() == ""
  assert dt.yaxis_title_from_operator("m") == "Forskel fra grundforløb"

  dt.set_language("en")

def test_plotting():
  """
  Plotting is difficult to unit test, but we can at least test that the functions run without errors.
  Visual inspection is required to verify that the plots are correct.
  """
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")

  dt.time(2025, 20260)
  df1 = dt.DataFrame(
    [s.qC, s.qG, s.qI, s.qX, s.qM],
    names=["Privat forbrug (qC)", "Offentligt forbrug (qG)", "Investeringer (qI)", "Eksport (qX)", "Import (qM)"]
  )
  fig1 = df1.plot()
  assert fig1 == df1.plot()
  
  df2 = dt.DataFrame(
    [s.qC, s.qG, s.qI, s.qX, s.qM],
    "pq",
    names=["Privat forbrug (qC)", "Offentligt forbrug (qG)", "Investeringer (qI)", "Eksport (qX)", "Import (qM)"]
  )
  fig2 = df2.plot(figure_size="document_small")
  
  df3 = dt.DataFrame(s.qY[s.s], names=list(s.s.texts))
  fig3 = df3.plot()
  assert dt.plot(s.qY[s.s], names=list(s.s.texts)) == fig3

  dt.write_image(fig1, "test1.png", scale=1)
  dt.write_image(fig2, "test2.png", scale=1)
  dt.write_image(fig3, "test3.png", scale=1)

  assert round(Image.open("test1.png").width / 72 * 2.54, 1) == 15.5, "Large figure is not the correct width (pixels × DPI)"
  assert round(Image.open("test2.png").width / 72 * 2.54, 1) == 7.6, "Small figure is not the correct width (pixels × DPI)"

def test_figure_size_presets():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame(s.qY, "m")

  assert round(df.plot(figure_size="slide_large").layout.width / 72 * 2.54, 1) == 15.5
  assert round(df.plot(figure_size="slide_small").layout.width / 72 * 2.54, 1) == 7.6
  assert round(df.plot(figure_size="document_large").layout.width / 72 * 2.54, 1) == 15.5
  assert round(df.plot(figure_size="document_small").layout.width / 72 * 2.54, 1) == 7.6
  assert df.plot().layout.width == df.plot(figure_size="document_large").layout.width

def test_svg_size_is_written_in_cm():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  fig = dt.DataFrame(s.qY, "m").plot()
  dt.write_image(fig, "test.svg", scale=1)
  root = ET.parse("test.svg").getroot()
  assert root.attrib["width"].endswith("cm")
  assert root.attrib["height"].endswith("cm")
  assert round(float(root.attrib["width"].removesuffix("cm")), 1) == 15.5

def test_pdf_size_matches_figure_width():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  fig = dt.DataFrame(s.qY, "m").plot(figure_size="document_small")
  dt.write_image(fig, "test.pdf")
  with open("test.pdf", "rb") as file:
    media_box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", file.read()).group(1).split()
  assert round((float(media_box[2]) - float(media_box[0])) / 72 * 2.54, 1) == 7.6

@pytest.mark.skipif(
  not all(os.path.isdir(path) for path in TEMPLATE_PDF_DIRS),
  reason="Template directories are not available.",
)
def test_write_template_pdf_figures():
  dt.time(2026, 2050)
  index = [2026, 2027, 2028]
  make_df = lambda names: dt.DataFrame([
    pd.Series([i, i + 0.1, i + 0.2], index=index, name=name)
    for i, name in enumerate(names, start=1)
  ])
  make_numbered_df = lambda label, count: make_df([f"{label} line {i}" for i in range(1, count + 1)])
  short_labels = ["GDP", "Consumption", "Government", "Investments", "Exports", "Imports", "Employment", "Wages", "Prices"]
  age_profile_index = pd.MultiIndex.from_product([
    list(range(18, 101)),
    [2026, 2035, 2050],
  ])
  supply_balance_labels = [
    "GDP relative to baseline (qBNP)",
    "Private consumption relative to baseline GDP (qC/qBNP)",
    "Public consumption relative to baseline GDP (qG/qBNP)",
    "Investments relative to baseline GDP (qI/qBNP)",
    "Exports relative to baseline GDP (qX/qBNP)",
    "Imports relative to baseline GDP (qM/qBNP)",
  ]
  figures = {
    "small_figure_1.pdf": make_numbered_df("Small", 3).plot(
      figure_size="document_small",
      layout=dict(
        yaxis_title_text="Small figure y-axis",
        xaxis_title_text="Year",
      ),
    ),
    "small_figure_2.pdf": make_numbered_df("Small", 6).plot(
      figure_size="document_small",
      layout=dict(yaxis_title_text="Small dashed figure y-axis"),
      alternating_dash="dot",
    ),
    "small_figure_3.pdf": make_numbered_df("Small", 7).plot(
      figure_size="document_small",
      layout=dict(yaxis_title_text="Small 7-line figure y-axis"),
    ),
    "small_figure_4.pdf": make_df(short_labels).plot(
      figure_size="document_small",
      layout=dict(yaxis_title_text="Many short legend labels"),
      alternating_dash="dot",
    ),
    "large_figure.pdf": make_df(supply_balance_labels).plot(
      figure_size="document_large",
      layout=dict(yaxis_title_text="Supply balance quantities"),
    ),
    "age_2d.pdf": dt.age_figure_2d(
      pd.Series(
        [a * 2.0 + y * 0.05 for a, y in age_profile_index],
        index=age_profile_index,
        name="Example variable",
      ),
      years=[2026, 2035, 2050],
      figure_size="document_small",
      layout=dict(yaxis_title_text="Age profile (example)"),
    ),
  }
  for output_dir in TEMPLATE_PDF_DIRS:
    for file_name, fig in figures.items():
      dt.write_image(fig, os.path.join(output_dir, file_name))

def test_x_axis_title_reserves_space_above_colored_legend():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame([s.qC, s.qG], names=["a", "b"])
  fig_base = df.plot()
  fig_x = df.plot(layout=dict(xaxis_title_text="Year"))
  assert fig_x.layout.margin.b > fig_base.layout.margin.b
  assert fig_x.layout.annotations[-2].yshift < fig_base.layout.annotations[-2].yshift

def test_x_axis_title_reserves_space_with_reserve_legend():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame([s.qC, s.qG], names=["a", "b"])
  fig_base = df.plot(colored_legend=False)
  fig_x = df.plot(colored_legend=False, layout=dict(xaxis_title_text="Year"))
  assert fig_x.layout.margin.b > fig_base.layout.margin.b

def test_legend_is_centered_below_plot():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame([s.qC, s.qG], names=["a", "b"])
  fig = df.plot()
  assert fig.layout.showlegend is False
  assert all(trace.showlegend is False for trace in fig.data)
  fs = fig.layout.legend.font.size or fig.layout.font.size or 10
  inset = 0.5 * fs / dt.legend_width(fig)
  assert [annotation.x for annotation in fig.layout.annotations[-2:]] == [inset, 0.5 + inset]
  assert all(annotation.xanchor == "left" for annotation in fig.layout.annotations[-2:])
  assert fig.layout.annotations[-2].yshift < -2 * fig.layout.font.size

def test_showlegend_false_hides_custom_timeseries_legend():
  dt.time(2026, 2027)
  index = [2026, 2027]
  df = dt.DataFrame([
    pd.Series([1, 2], index=index, name="a"),
    pd.Series([2, 3], index=index, name="b"),
  ])

  for fig in [df.plot(layout=dict(showlegend=False)), df.plot(showlegend=False)]:
    assert len(fig.data) == 2
    assert fig.layout.showlegend is False
    assert not fig.layout.annotations

def test_showlegend_false_hides_custom_age_legend():
  dt.time(2026, 2027)
  age_index = pd.MultiIndex.from_product([[18, 19], [2026, 2027]])
  series = pd.Series(np.arange(len(age_index)), index=age_index, name="a")
  fig = dt.age_figure_2d(series, years=[2026, 2027], showlegend=False)

  assert len(fig.data) == 2
  assert fig.layout.showlegend is False
  assert not fig.layout.annotations

def test_legend_uses_same_columns_on_uneven_rows():
  df = pd.DataFrame({
    "a": [1, 2],
    "b": [2, 3],
    "c": [3, 4],
  })
  fig = df.plot()
  fig.update_layout(**dt.figure_layouts["document_large"])
  fig = dt.colored_text_legend(fig, columns=2)
  fs = fig.layout.legend.font.size or fig.layout.font.size or 10
  inset = 0.5 * fs / dt.legend_width(fig)
  assert [annotation.x for annotation in fig.layout.annotations[-3:]] == [inset, 0.5 + inset, inset]

def test_legend_labels_use_trace_colors():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame([s.qC, s.qG], names=["a", "b"])
  fig = df.plot()
  legend_annotations = fig.layout.annotations[-2:]
  assert legend_annotations[0].font.color == dt.dream_colors_rgb["DREAM"]
  assert legend_annotations[1].font.color == dt.dream_colors_rgb["REFORM"]
  assert not fig.layout.shapes

def test_dashed_legend_labels_include_line_samples():
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame([s.qC, s.qG], names=["a", "b"])
  fig = df.plot(alternating_dash="dot")
  legend_annotations = fig.layout.annotations[-2:]
  legend_shapes = fig.layout.shapes[-2:]
  assert [shape.line.dash for shape in legend_shapes] == ["solid", "dot"]
  assert [annotation.xanchor for annotation in legend_annotations] == ["left", "left"]
  assert all(shape.x1 < annotation.x for shape, annotation in zip(legend_shapes, legend_annotations))
  assert all(trace.showlegend is False for trace in fig.data)

def test_legend_labels_are_wrapped():
  df = pd.DataFrame({
    "A very long legend label that should wrap before it overlaps the next legend label": [1, 2],
    "Another very long legend label that should wrap before it overlaps the previous label": [2, 1],
  })
  fig = dt.wrap_legend_labels(df.plot(), width=48)
  assert "<br>" in fig.data[0].name
  assert "<br>" in fig.data[1].name

def test_auto_legend_labels_use_available_width():
  index = [2026, 2027]
  labels = [
    "GDP relative to baseline (qBNP)",
    "Private consumption relative to baseline GDP (qC/qBNP)",
    "Public consumption relative to baseline GDP (qG/qBNP)",
    "Investments relative to baseline GDP (qI/qBNP)",
    "Exports relative to baseline GDP (qX/qBNP)",
    "Imports relative to baseline GDP (qM/qBNP)",
  ]
  df = dt.DataFrame([pd.Series([i, i + 0.1], index=index, name=name) for i, name in enumerate(labels)])
  fig = df.plot(figure_size="document_large")
  assert all("<br>" not in annotation.text for annotation in fig.layout.annotations[-len(labels):])

def test_auto_legend_spreads_short_labels_across_columns():
  index = [2026, 2027]
  labels = ["GDP", "Consumption", "Government", "Investments", "Exports", "Imports"]
  df = dt.DataFrame([pd.Series([i, i + 0.1], index=index, name=name) for i, name in enumerate(labels)])
  fig = df.plot(figure_size="document_large")
  assert len({annotation.x for annotation in fig.layout.annotations[-len(labels):]}) > 1

def test_legend_column_count_includes_line_sample_width():
  """Dashed traces reserve segment + gap width like legend_entry_layout; column fit must match."""
  dt.time(2026, 2050)
  index = [2026, 2027, 2028]
  names = ["GDP", "Consumption", "Government", "Investments", "Exports", "Imports", "Employment", "Wages", "Prices"]
  df = dt.DataFrame([
    pd.Series([i, i + 0.1, i + 0.2], index=index, name=name)
    for i, name in enumerate(names, start=1)
  ])
  fig_plain = df.plot(figure_size="document_small")
  fig_dash = df.plot(figure_size="document_small", alternating_dash="dot")
  fs = fig_plain.layout.legend.font.size or fig_plain.layout.font.size or 10
  assert dt.legend_column_count(fig_plain, list(fig_plain.data), fs) >= dt.legend_column_count(fig_dash, list(fig_dash.data), fs)

def test_legend_space_preserves_chart_area():
  df = pd.DataFrame({
    "A very long legend label that should wrap before it overlaps the next legend label": [1, 2],
    "Another very long legend label that should wrap before it overlaps the previous label": [2, 1],
  })
  fig = dt.wrap_legend_labels(df.plot(), width=48)
  fig.update_layout(height=300, margin_b=30, legend=dict(font=dict(size=10)))
  fig = dt.reserve_legend_space(fig, columns=1)
  assert fig.layout.margin.b > 30
  assert fig.layout.height - fig.layout.margin.b == 270
  
def test_yaxis_title():
  dt.set_language("en")
  dt.REFERENCE_DATABASE = dt.Gdx("test.gdx")
  s = dt.Gdx("test.gdx")
  df = dt.DataFrame(s.qY, "m")
  assert df.plot().layout.yaxis.title.text == dt.yaxis_title_from_operator("m")
  assert df.plot(layout=dict(yaxis_title="y-axis title")).layout.yaxis.title.text == "y-axis title"
  assert df.plot(layout=dict(yaxis_title="y-axis title"), horizontal_yaxis_title=True).layout.yaxis.title.text == ""

def test_yaxis_title_font_matches_figure_size_for_timeseries_and_age():
  expected = dt.figure_layouts["document_small"]["yaxis"]["title_font"]["size"]
  years = [2026, 2027, 2028]
  ts = pd.Series([1.0, 1.2, 1.4], index=years, name="series")
  age_index = pd.MultiIndex.from_product([[18, 19, 20], years])
  age = pd.Series(np.arange(len(age_index), dtype=float), index=age_index, name="series")
  ts_fig = dt.DataFrame(ts).plot(figure_size="document_small", layout=dict(yaxis_title="Timeseries y-axis"))
  age_fig = dt.age_figure_2d(age, years=years, figure_size="document_small", layout=dict(yaxis_title="Age y-axis"))
  assert ts_fig.layout.yaxis.title.font.size == expected
  assert age_fig.layout.yaxis.title.font.size == expected
  