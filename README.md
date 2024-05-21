# DREAM-tools
A collection of tools used by [DREAM, Danish Research Institute for Economic Analysis and Modelling](https://dreamgruppen.dk/).

## Gams-Pandas
A wrapper around the [GAMS Python api](https://www.gams.com/latest/docs/API_PY_OVERVIEW.html) to move smoothly between GAMS and [Pandas](https://pandas.pydata.org/).
GAMS parameters are represented as Pandas Series, using a MultiIndex in cases of multiple sets.
The level value of variables are represented in the same way. GAMS sets are represented as Pandas Index to facilitate easy use of label based indexing.

## Excel-gdx
Access GAMS gdx files directly from Excel using [xlwings](https://www.xlwings.org/).

## Plotting
DREAM-tools contains a number of plotting features based on [plotly](https://plot.ly/python/).

## gamY
A pre-processor for GAMS files implementing a number of additional features.

