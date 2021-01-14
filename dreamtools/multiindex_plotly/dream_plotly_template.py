import plotly.graph_objects as go

dream = {
    "layout": {
        "colorway": [
            "rgb(245,82,82)",
            "rgb(20,175,166)",
            "rgb(255,155,75)",
            "rgb(92,210,114)",
            "rgb(66,180,224)",
            "rgb(188,173,221)",
            "rgb(0,95,151)",
            "rgb(137,48,112)",
            "rgb(70,70,76)",
        ],
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