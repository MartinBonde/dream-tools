import xlwings as xw
import logging
from gdx import Gdx
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ACTIVE_GDX_PATH = None


@xw.func
@xw.arg('path', doc='Path of GDX file.')
def set_gdx_path(path):
  """Set file path of active GDX file."""
  global ACTIVE_GDX_PATH
  ACTIVE_GDX_PATH = os.path.abspath(path)
  logger.info(f"ACTIVE_GDX_PATH set to {ACTIVE_GDX_PATH}")
  return f"Active gdx set to {ACTIVE_GDX_PATH}"


@xw.func
def get_gdx_path():
  """Get file path currently active GDX file."""
  return ACTIVE_GDX_PATH


@xw.arg('request', doc='Name of variable.')
@xw.arg('year', doc='Optional. Year to get data.')
@xw.arg('gdx_path', doc='Optional. File path of gdx file to get data from.')
@xw.func
@xw.ret(header=True, index=True, expand='table')
def gdx(request, year=None, gdx_path=None):
  if gdx_path is None:
    if ACTIVE_GDX_PATH is None:
      return "#ERROR! Use set_gdx_path to choose a gdx file."
    else:
      gdx_path = ACTIVE_GDX_PATH
  gdx = Gdx(gdx_path)
  var_name, records = parse_request(gdx, request)
  if var_name not in gdx:
    return "#VALUE!"
  result = gdx[var_name]
  if records:
    result = result.loc[records]
  if year:
    result = result.xs(str(round(year)), level="t")
  if len(result) == 1:
    result = result.values
  return result


# @xw.func
# @xw.ret(header=False, index=False, expand='table')
# def test(x):
#   # return str(gams.GamsWorkspace(system_directory="C:\\GAMS\\win64\\28.2", debug=3))
#   # return str(gams.GamsWorkspace())
#   return(str(x))


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
    return [domain]


