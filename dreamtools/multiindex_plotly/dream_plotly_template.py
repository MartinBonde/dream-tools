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
  "F55252", # DREAM
  "42B4E0", # REFORM
  "5CD272", # Grøn REFORM
  "FF9B4B", # SMILE
  "14AFA6", # MAKRO
  "BCADDD", # Plum
  "005F97", # Dark blue
  "893070", # Maroon
  "46464C", # Dark gray
  "E6E6E8", # Light gray
]

DPI = 72
DPCM = DPI / 2.54
FONT_FAMILY = "Hind, Arial, sans-serif"

def cm(x):
  return x * DPCM

def figure_layout(
  width_cm,
  height_cm,
  font_size,
  title_font_size,
  tick_font_size,
  margin_l_cm=1.5,
  margin_r_cm=1.0,
  margin_t_cm=1.0,
  margin_b_cm=1.0,
):
  return dict(
    width=cm(width_cm),
    height=cm(height_cm),
    margin_l=cm(margin_l_cm),
    margin_r=cm(margin_r_cm),
    margin_t=cm(margin_t_cm),
    margin_b=cm(margin_b_cm),
    title_font_size=title_font_size,
    font=dict(size=font_size, family=FONT_FAMILY),
    legend=dict(
      title_text="",
      orientation="v",
      yref="container",
      y=0,
      yanchor="bottom",
      x=0.5,
      xanchor="center",
      font=dict(size=font_size),
    ),
    xaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=tick_font_size)),
    yaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=tick_font_size)),
  )

figure_layouts = {
  "slide_large": figure_layout(
    24.0, 13.5, font_size=16, title_font_size=18, tick_font_size=14,
    margin_l_cm=2.0, margin_r_cm=1.2, margin_t_cm=1.2, margin_b_cm=1.2,
  ),
  "slide_small": figure_layout(
    11.5, 8.0, font_size=13, title_font_size=15, tick_font_size=12,
    margin_l_cm=1.6, margin_r_cm=0.8, margin_t_cm=1.0, margin_b_cm=1.0,
  ),
  "document_large": figure_layout(
    15.5, 10.0, font_size=10, title_font_size=10, tick_font_size=9,
  ),
  "document_small": figure_layout(
    7.6, 7.0, font_size=8, title_font_size=9, tick_font_size=8,
    margin_l_cm=1.2, margin_r_cm=0.7, margin_t_cm=0.8, margin_b_cm=0.8,
  ),
}

def create_dream_template(trace_count, items_per_row=2):
  """
  Create a Plotly template with DREAM colors and ordinary Plotly layout behavior.
  
  The trace_count and items_per_row arguments are kept for backwards compatibility.
  :return: Plotly template.
  """
  dream_layout = dict(
    colorway=list(dream_colors_rgb.values()),
    title_font_size=10,
    legend=dict(
      title_text="",
      orientation="v",
      yref="container",
      y=0,
      yanchor="bottom",
      x=0.5,
      xanchor="center",
    ),
    font=dict(size=10, family=FONT_FAMILY),
    margin=dict(
      l=1.5 * DPCM,
      r=1.0 * DPCM,
      t=1 * DPCM,
      b=1 * DPCM,
    ),
    width=15.5 * DPCM,
    height=10 * DPCM,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
      title_font=dict(size=10),
      tickfont=dict(size=9),
      showgrid=True,
      gridcolor=dream_colors_rgb["Light gray"],
      ticks="outside",
      ticklen=4,
      showline=True,
      mirror=True,
      zeroline=False,
      automargin=True,
    ),
    yaxis=dict(
      title_font=dict(size=10),
      tickfont=dict(size=9),
      showgrid=True,
      gridcolor=dream_colors_rgb["Light gray"],
      ticks="outside",
      ticklen=4,
      showline=True,
      mirror=True,
      zeroline=False,
      automargin=True,
    ),
  )

  dream_template_data = dict(
    scatter=[
      dict(
        line=dict(width=1),
      )
    ]
  )

  dream_template = go.layout.Template(
    layout=dream_layout,
    data=dream_template_data,
  )
  
  return dream_template
