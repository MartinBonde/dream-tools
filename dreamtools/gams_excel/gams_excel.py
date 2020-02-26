import xlwings as xw
import logging

from dreamtools import unstack_multiseries

from dreamtools.gams_pandas import Gdx
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ACTIVE_GDX = None


@xw.func
@xw.arg('path', doc='Path of GDX file.')
def set_gdx_path(path):
  """Set file path of active GDX file."""
  global ACTIVE_GDX
  ACTIVE_GDX = Gdx(path)
  logger.info(f"ACTIVE_GDX set to {ACTIVE_GDX.abs_path}")
  return f"Active gdx set to {ACTIVE_GDX.abs_path}"


@xw.func
def get_gdx_path():
  """Get file path of currently active GDX file."""
  return getattr(ACTIVE_GDX, "abs_path", None)


@xw.arg('request', doc='Name of variable.')
@xw.arg('year', doc='Optional. Year to get data.')
@xw.arg('gdx_path', doc='Optional. File path of gdx file to get data from.')
@xw.arg('array_results', doc='Optional, default=False. If True, the function can return an array with values from multiple years or array series elements.')
@xw.func
@xw.ret(header=True, index=True, expand='table')
def gdx(request, year=None, gdx_path=None, array_results=False):
  """
  Look up the request in a gdx file.
  A request uses Gekko syntax, e.g. "qBNP", "qY[tje]", or "qY[#s]".
  If no year is specified, an array is returned with values for all available years.
  If no gdx_path is specified, use the global path set using set_gdx_path
  """
  if gdx_path is not None:
    gdx = Gdx(gdx_path)
  elif ACTIVE_GDX is not None:
    gdx = ACTIVE_GDX
  else:
    return "#ERROR! Use set_gdx_path to choose a gdx file."

  var_name, records = parse_request(gdx, request)
  if var_name not in gdx:
    return "#VALUE!"
  result = gdx[var_name]
  if records:
    result = result.loc[records]
  result = unstack_multiseries(result, keep_axis_index=-1)
  if year:
    result = result.loc[year].values
  if len(result) > 1 and not array_results:
    return "#VALUE!"
  else:
    return result


def parse_request(gdx, name):
  if name.count("[") == 1 and name.endswith("]"):
    var_name = name.split("[")[0]
    domains = name.split("[")[1][:-1].split(",")
    return var_name, tuple(parse_domain(gdx, domain) for domain in domains)
  else:
    return name, []


def parse_domain(gdx, domain):
  if domain.startswith("#"):
    return ["".join(element) for element in gdx[domain[1:]].index]  # "".join(element) 'de-tuples' the index element
  else:
    try:
      return [int(domain)]
    except ValueError:
      return [domain]



# @xw.func
# @xw.ret(header=False, index=False, expand='table')
# def test(x):
#   # return str(gams.GamsWorkspace(system_directory="C:\\GAMS\\win64\\28.2", debug=3))
#   # return str(gams.GamsWorkspace())
#   return(str(x))
