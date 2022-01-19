import os

import gams

from .gams_pandas import GamsPandasDatabase, logger


class Gdx(GamsPandasDatabase):
  """Wrapper that opens a GDX file as a GamsPandasDatabase."""

  def __init__(self, file_path, workspace=None, sparse=True):
    if not file_path:
      import easygui
      file_path = easygui.fileopenbox("Select reference gdx file", filetypes=["*.gdx"])
      if not file_path:
        raise ValueError("No file path was provided")
    if not os.path.splitext(file_path)[1]:
      file_path = file_path + ".gdx"
    self.abs_path = os.path.abspath(file_path)
    logger.info(f"Open GDX file from {self.abs_path}.")
    if workspace is None:
      workspace = gams.GamsWorkspace()
    database = workspace.add_database_from_gdx(self.abs_path)
    super().__init__(database, sparse=sparse)

  def export(self, path=None, relative_path=True):
    """
    Save changes to database and export database to GDX file.
    If no path is specified, the path of the linked GDX file is used, overwriting the original.
    Use relative_path=True to use export to a path relative to the directory of the originally linked GDX file.
    """
    if path is None:
      path = self.abs_path
    elif relative_path:
      path = os.path.join(os.path.dirname(self.abs_path), path)
    logger.info(f"Export GDX file to {path}.")
    super().export(path)

  def __repr__(self):
    return f"dreamtools.gams_pandas.gams_pandas.Gdx from {self.abs_path}"