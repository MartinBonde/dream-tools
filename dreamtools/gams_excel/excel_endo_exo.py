"""
Transform Excel work sheet into GAMS code to perform endo exo operations and change variable levels.
"""
import os
import sys
import pandas as pd
import xlwings as xw


def main():
  #  Read execution parameters
  workbook_path = os.path.abspath(sys.argv[1])
  assert os.path.splitext(workbook_path)[1].lower() in [".xls", ".xlsx", ".xlsm"], \
    f"{workbook_path} is not an Excel workbook"

  output_path = os.path.abspath(sys.argv[2])

  # Read Excel file
  print(f"Read endo_exo sheet from {workbook_path}")
  app = xw.App(visible=False)
  workbook = xw.Book(workbook_path)
  endo_exo = workbook.sheets["endo_exo"].range("B4:CK5000").options(pd.DataFrame, index=False).value
  app.quit()

  # Cast years in column header to ints
  header = list(endo_exo)
  year_columns_index = 4
  header[year_columns_index:] = [int(i) for i in header[year_columns_index:]]
  endo_exo.columns = header

  endo_exo_strings = endo_exo.apply(row_to_GAMS, axis=1, result_type="reduce", year_columns_index=year_columns_index)

  with open(output_path, "w") as file:
    print(f"Write to {output_path}")
    file.write("\n".join(endo_exo_strings))


def row_to_GAMS(row, year_columns_index, comment_char="#"):
  """Return GAMS code from row"""
  endo_exo_strings = []
  for interval in split_intervals(row[year_columns_index:]):
    if interval and not row['#']:
      interval_string = f"{interval[0]} <= t.val and t.val <= {interval[1]}"
      if row.endo:
        endo_exo_strings += [f"UNFIX {row.endo}$({interval_string});"]
      if row.exo:
        for t in interval:
          val = row.loc[t]
          name, elements = split_name(row.exo)
          var_input = row.exo.strip()
          if var_input.endswith("]")
            var = f"{var_input[:-1]},'{t}']"
            lagged = f"{var_input[:-1]},'{t-1}']"
          else
            var = f"{var_input}['{t}']"
            lagged = f"{var_input}['{t-1}']"
          if row.printcode == "dummy":
            pass
          elif row.printcode == "p":
            endo_exo_strings += [f"{var}$(t.val = {t}) = (1 + {val}/100) * ;"]
          elif row.printcode == "n":
            endo_exo_strings += [f"{var}$() = {val};"]
      else # Transfer comments
      endo_exo_strings += [f"{comment_char} {i}" for i in row.dropna()]

  return "\n".join(endo_exo_strings)

def split_name(input):
  """
  Split Gekko syntax name
  Return symbol name and a list of elements
  Example:
    >>> split_name("qI[#i,CON]")
    "qI", ["#i", "'CON'"]
  """
  if "[" in input:
    name, elements = input.strip().split("[", 1)
    elements = elements[:-1].split(",")
    elements = [e[1:] if (e[0] == "#") else f"'{e}'" for e in elements]
  else:
    name = input
    elements = []
  return name, elements


def split_intervals(series):
  interval = []
  for year in series.dropna().index:
    if (year - 1) not in interval:
      if interval:
        yield interval
        interval = []
    interval += [year]
  yield interval


def error(msg):
  exit(f"ERROR! {msg}")


if __name__ == "__main__":
  main()
