import plotly.graph_objects as go

dream_colors_rgb = {
  "DREAM": "rgb(245,82,82)",
  "REFORM": "rgb(66,180,224)",
  "Grøn REFORM": "rgb(92,210,114)",
  "SMILE": "rgb(255,155,75)",
  "MAKRO": "rgb(20,175,166)",
  "Plum": "rgb(188,173,221)",
  "Dark blue": "rgb(0,95,151)",
  "Maroon": "rgb(137,48,112)",
  "Dark gray": "rgb(70,70,76)",
  "Light gray": "rgb(230,230,232)",
}

dream_colors_hex = [
  "F55252",
  "14AFA6",
  "FF9B4B",
  "5CD272",
  "42B4E0",
  "BCADDD",
  "005F97",
  "893070",
  "46464C",
  "E6E6E8",
]

DPI = 96
DPCM = DPI / 2.54

small_figure_layout = dict(
  width = 7.6 * DPCM,
  margin_l = 1.5 * DPCM,
  margin_r = 0.5 * DPCM,
  margin_t = 1 * DPCM,
)

large_figure_layout = dict(
  width = 15.5 * DPCM,
  margin_l = 1.5 * DPCM,
  margin_r = 0.5 * DPCM,
  margin_t = 1 * DPCM,
)

dream_layout = dict(

  colorway = list(dream_colors_rgb.values()),

  title_font_size = 10,

  legend_orientation = "h",
  legend_yanchor = "top",
  legend_y = -0.15,
  legend_xanchor = "center",
  legend_x = 0.5,

  font_color = "black",
  font_family = "Hind",
  font_size = 10,

  xaxis_ticks = "outside",
  xaxis_ticklen = 10,
  xaxis_showline = True,
  xaxis_linecolor = "black",
  xaxis_automargin = False,
  xaxis_title_standoff = 5,
  xaxis_title_font_size = 10,

  yaxis_ticks = "outside",
  yaxis_ticklen = 10,
  yaxis_showline = True,
  yaxis_linecolor = "black",
  yaxis_automargin = False,
  yaxis_tickmode = "auto",
  yaxis_title_font_size = 10,

  xaxis_gridcolor = dream_colors_rgb["Light gray"],
  yaxis_gridcolor = dream_colors_rgb["Light gray"],

  **large_figure_layout
)

dream_template_data = dict(
  scatter = [
    dict(
      line = dict(width = 4,),
    ),
  ],
)
dream_template_data["scattergl"] = dream_template_data["scatter"]

dream_template = go.layout.Template(layout=dream_layout, data=dream_template_data)
