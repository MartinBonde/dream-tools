import plotly.graph_objects as go

dream_colors_rgb = [
  "rgb(245,82,82)",   # Red (DREAM)
  "rgb(20,175,166)",  # Pretrolium (MAKRO)
  "rgb(255,155,75)",  # Orange (SMILE)
  "rgb(92,210,114)",  # Green (Grøn REFORM)
  "rgb(66,180,224)",  # Blue (REFORM)
  "rgb(188,173,221)", # Plum
  "rgb(0,95,151)",    # Dark blue
  "rgb(137,48,112)",  # Maroon
  "rgb(70,70,76)",    # Dark gray
  "rgb(230,230,232)", # Light gray
]

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

dream = {
  "layout": {
    "colorway": dream_colors_rgb,
    "legend": {
      "yanchor": "bottom",
      "xanchor": "center",
      "orientation": "h",
      "y": -0.5,
      "x": 0.5,
    },
    "font": {"color": "rgb(0,0,0)", "family": "Hind", "size": 14},

    "xaxis": {
      "ticks": "outside",
      "ticklen": 10,
      "showline": True,
      "linecolor": "black",
      "automargin": False,
      "title_standoff": 0,
      "title_font_size": 14,
    },

    "yaxis": {
      "ticks": "outside",
      "ticklen": 10,
      "showline": True,
      "linecolor": "black",
      # "tickangle": -90,
      "automargin": False,
      "tickmode": "auto",
    },

    "plot_bgcolor": "rgb(230,230,232)",
    "xaxis_gridcolor": "white",
    "yaxis_gridcolor": "white",

    # "paper_bgcolor": "rgb(230,230,232)",
  },

  "data": {
    "scatter": [{
      "line": {"width": 4},
    }],
  }
}

dream["data"]["scattergl"] = dream["data"]["scatter"]

dream_template = go.layout.Template(**dream)