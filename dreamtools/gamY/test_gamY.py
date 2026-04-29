import importlib
import re

import pytest


gamy = importlib.import_module("dreamtools.gamY.gamY")


@pytest.fixture(autouse=True)
def reset_gamy_settings():
  settings = {
    name: getattr(gamy, name)
    for name in [
      "leave_env_variables_for_gams",
      "automatic_additive_residuals_prefix",
      "automatic_multiplicative_residuals_prefix",
      "error_on_missing_label",
      "variable_equation_prefix",
      "automatic_dummy_suffix",
    ]
  }
  yield
  for name, value in settings.items():
    setattr(gamy, name, value)


def expand(tmp_path, text):
  file_path = tmp_path / "model.gms"
  file_path.write_text(text, encoding="utf-8")
  precompiler = gamy.Precompiler(file_path)
  return precompiler(), precompiler


def assert_contains_line(text, expected):
  assert re.search(rf"^\s*{re.escape(expected)}\s*$", text, re.MULTILINE)


def test_group_variants_can_define_symbols_and_be_looped_over(tmp_path):
  text = """
  $GROUP G_vars
    x[t] "Quantity" 1,
    y "Scalar";

  $PGROUP G_pars
    p[t] "Price parameter";

  $SETGROUP G_sets
    tt "Time set";

  $LOOP G_vars:
    saved_{name}{sets} = {name}.L{sets};
  $ENDLOOP

  $LOOP G_pars:
    par_{name}{sets} = {name}{sets};
  $ENDLOOP

  $LOOP G_sets:
    set_{name}{sets} = yes;
  $ENDLOOP
  """

  output, precompiler = expand(tmp_path, text)

  assert "Variable x[t] \"Quantity\" //;" in output
  assert re.search(r"x\.L\[t\]\s*=\s*1;", output)
  assert "Parameter p[t] \"Price parameter\" //;" in output
  assert "Set tt \"Time set\" //;" in output
  assert_contains_line(output, "saved_x[t] = x.L[t];")
  assert_contains_line(output, "saved_y = y.L;")
  assert_contains_line(output, "par_p[t] = p[t];")
  assert_contains_line(output, "set_tt = yes;")
  assert set(precompiler.groups["G_vars"]) == {"x", "y"}
  assert set(precompiler.par_groups["G_pars"]) == {"p"}
  assert set(precompiler.set_groups["G_sets"]) == {"tt"}


def test_display_all_ignores_group_conditions(tmp_path):
  text = """
  $GROUP G_vars
    x[t]$(t.val > 2020) "Quantity";

  $DISPLAY G_vars;
  $DISPLAY_ALL G_vars;
  """

  output, _ = expand(tmp_path, text)

  assert "report__t(t, 'x')$(t.val > 2020) = x.L[t] + 0;" in output
  assert "report__t(t, 'x')$(1) = x.L[t] + 0;" in output


def test_group_subset_syntax_is_equivalent_to_dollar_condition(tmp_path):
  text = """
  $GROUP G_all
    x[t] "Quantity";

  $GROUP G_subset
    x[tx];

  $GROUP G_dollar
    x[t]$(tx[t]);

  $LOOP G_subset:
    subset_{name}{sets}${conditions} = {name}.L{sets};
  $ENDLOOP
  """

  output, precompiler = expand(tmp_path, text)

  assert precompiler.groups["G_subset"]["x"].sets == "[t]"
  assert precompiler.groups_conditions["G_subset"]["x"] == "(tx[t])"
  assert precompiler.groups_conditions["G_subset"] == precompiler.groups_conditions["G_dollar"]
  assert_contains_line(output, "subset_x[t]$(tx[t]) = x.L[t];")


def test_block_can_map_equations_to_endogenous_group_and_add_dummy_condition(tmp_path):
  gamy.automatic_dummy_suffix = "_dummy"
  text = """
  $GROUP G_endo
    q[t] "Quantity";

  $BLOCK B_market G_endo $(tx0[t])
    q[t]$(q_active[t]).. q[t] =E= demand[t];
  $ENDBLOCK
  """

  output, precompiler = expand(tmp_path, text)

  assert "SET q_dummy[t];" in output
  assert "q_dummy[t]).." in output
  assert "EQUATION q[t];" in output
  assert precompiler.blocks["B_market"]["q"].conditions == "$(((tx0[t]) and (q_active[t])) and q_dummy[t])"


def test_block_can_add_automatic_adjustment_terms(tmp_path):
  gamy.automatic_additive_residuals_prefix = "j_"
  gamy.automatic_multiplicative_residuals_prefix = "jr_"
  text = """
  $GROUP G_endo
    q[t] "Quantity";

  $BLOCK B_market G_endo
    q[t].. q[t] =E= demand[t];
  $ENDBLOCK
  """

  output, _ = expand(tmp_path, text)

  assert "Variable jr_q[t] 'Multiplicative adjustment term in equations for q' //;" in output
  assert "Variable j_q[t] 'Additive adjustment term in equations for q' //;" in output
  assert re.search(r"\(\(q\[t\] \+ j_q\[t\]\) \* \(1\+jr_q\[t\]\)\)\s*=E=\s*demand\[t\];", output)


def test_block_retains_code_that_is_not_an_equation(tmp_path):
  text = """
  $BLOCK B_market
    E_q[t].. q[t] =E= demand[t];
    abort$(card(t) = 0) "Missing time set";
    E_p[t].. p[t] =E= 1;
  $ENDBLOCK
  """

  output, precompiler = expand(tmp_path, text)

  assert set(precompiler.blocks["B_market"]) == {"E_q", "E_p"}
  assert 'abort$(card(t) = 0) "Missing time set";' in output


def test_solve_macro_builds_temporary_model(tmp_path):
  text = """
  $BLOCK B_market
    E_q[t].. q[t] =E= demand[t];
  $ENDBLOCK

  $SOLVE B_market;
  """

  output, precompiler = expand(tmp_path, text)

  assert "MODEL temp_model_0 /" in output
  assert "E_q" in precompiler.blocks["temp_model_0"]
  assert "Solve temp_model_0 using CNS;" in output


def test_eval_python_and_exec_python_macros(tmp_path):
  text = """
  $ExecPython
  self.globals["scenario"] = "base"
  $EndExecPython

  set s /%scenario%/;

  $EvalPython
  "scalar n /" + str(2 + 3) + "/;"
  $EndEvalPython
  """

  output, _ = expand(tmp_path, text)

  assert "set s /base/;" in output
  assert "scalar n /5/;" in output


@pytest.mark.parametrize(
  ("expression", "expected"),
  [
    ("(foo)(bar)", False),
    ("((foo)(bar))", True),
    ("[[foo]{bar}]", True),
  ],
)
def test_is_enclosed(expression, expected):
  assert gamy.is_enclosed(expression) is expected


def test_combine_conditions_removes_duplicates_and_supports_union():
  assert gamy.Precompiler.combine_conditions("$(a[t])", "$(b[t])", "$(a[t])") == "((a[t]) and (b[t]))"
  assert gamy.Precompiler.combine_conditions("$(a[t])", "$(b[t])", intersect=False) == "((a[t]) or (b[t]))"
