import plotly.graph_objects as go
import math

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

DPI = 96
DPCM = DPI / 2.54

large_figure_layout = dict(
  width = 15.5 * DPCM,
  # height = (plot_height + 3.8) * DPCM,
  margin_l = 1.5 * DPCM,
  margin_r = 0.5 * DPCM,
  margin_t = 1 * DPCM,
  # margin_b = 3.8 * DPCM,
)

small_figure_layout = dict(
  width = 7.6 * DPCM,
  # height = (plot_height + 5) * DPCM,
  margin_l = 1.5 * DPCM,
  margin_r = 0.5 * DPCM,
  margin_t = 1 * DPCM,
  # margin_b = 5 * DPCM,
)

def calculate_legend_height(trace_count, legend_item_height_cm=0.5, items_per_row=2):
  """
  Calculate the legend height based on the number of legend items,
  assuming a fixed number of items per row.
  
  :param trace_count: Number of traces (legend items) in the plot.
  :param legend_item_height_cm: Height per legend item in cm.
  :param items_per_row: Number of legend items per row.
  :return: Height of the legend in cm.
  """
  rows = math.ceil(trace_count / items_per_row)
  return rows * legend_item_height_cm

def calculate_figure_height(trace_count, base_height_cm=5, legend_item_height_cm=0.5, items_per_row=2):
  """
  Calculate the figure height based on the number of legend items.
  
  :param trace_count: Number of traces (legend items) in the plot.
  :param base_height_cm: Base height of the plot in cm.
  :param legend_item_height_cm: Height per legend item in cm.
  :param items_per_row: Number of legend items per row.
  :return: Total height of the figure in cm.
  """
  legend_height_cm = calculate_legend_height(trace_count, legend_item_height_cm, items_per_row)
  total_height_cm = base_height_cm + legend_height_cm
  return total_height_cm * DPCM

def create_dream_template(trace_count, items_per_row=2):
  """
  Create a Plotly template with dynamic height based on the number of traces.
  
  :param trace_count: Number of traces (legend items) in the plot.
  :param items_per_row: Number of legend items per row.
  :return: Plotly template.
  """
  figure_height = calculate_figure_height(trace_count, items_per_row=items_per_row)
  legend_height_cm = calculate_legend_height(trace_count, items_per_row=items_per_row)
  
  dream_layout = dict(
    colorway=list(dream_colors_rgb.values()),
    title_font_size=10,
    legend_orientation="h",
    legend_yanchor="top",
    legend_y=-legend_height_cm / DPCM,  # Position the legend below the plot
    legend_x=0.5,
    legend_xanchor="center",
    font=dict(size=10, family="Hind"),
    margin=dict(
      l=1.5 * DPCM,
      r=0.5 * DPCM,
      t=1 * DPCM,
      b=(legend_height_cm + 1) * DPCM  # Adjust bottom margin to accommodate the legend
    ),
    width=15.5 * DPCM,
    height=figure_height,
    xaxis=dict(
      title_font=dict(size=10),
      tickfont=dict(size=9),
      showgrid=True,
      ticks="outside",
      showline=True,
      mirror=True,
      zeroline=False,
    ),
    yaxis=dict(
      title_font=dict(size=10),
      tickfont=dict(size=9),
      showgrid=True,
      ticks="outside",
      showline=True,
      mirror=True,
      zeroline=False,
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

# # Example usage
# trace_count = 10  # Number of traces (legend items)
# fig = go.Figure()

# # Add some sample traces
# for i in range(trace_count):
#   fig.add_trace(go.Scatter(
#     x=[1, 2, 3],
#     y=[i, i + 1, i + 2],
#     mode='lines+markers',
#     name=f'Trace {i}'
#   ))

# # Apply the custom template
# fig.update_layout(template=create_dream_template(trace_count))

# fig.show()