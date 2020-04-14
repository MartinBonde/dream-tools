# DREAM-tools
A collection of tools used by [the Danish institute for economic modelling and forecasting, DREAM](http://dreammodel.dk).


## Gams-Pandas
A wrapper around the [GAMS Python api](https://www.gams.com/latest/docs/API_PY_OVERVIEW.html) to move smoothly between GAMS and [Pandas](https://pandas.pydata.org/).
GAMS parameters are represented as Pandas Series, using a MultiIndex in cases of multiple sets.
The level value of variables are represented in the same way. GAMS sets are represented as Pandas Index to facilitate easy use of label based indexing.

## Excel-gdx
Access GAMS gdx files directly from Excel using [xlwings](https://www.xlwings.org/).

## Plotting
DREAM-tools contains a number of plotting features based on [plotly](https://plot.ly/python/) and overwrites the plot method of Pandas DataFrames and Series to utilize the features by default.

## gamY
A pre-processor for GAMS files implementing a number of additional features.


## Gekko
A script is included to endogenize or exogenize variables in GAMS from [Gekko Timeseries and Modeling Software](http://t-t.dk/gekko/).