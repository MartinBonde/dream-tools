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


def test_macro_variables_set_eval_and_scope(tmp_path):
  text = """
  $SETGLOBAL terminal_year 2060
  $SETLOCAL scenario baseline
  $EVAL next_year 2059 + 1
  scalar t /%terminal_year%/;
  set s /%scenario%/;
  scalar n /%next_year%/;
  """

  output, precompiler = expand(tmp_path, text)

  assert "scalar t /2060/;" in output
  assert "set s /baseline/;" in output
  assert "scalar n /2060/;" in output
  assert precompiler.globals["terminal_year"] == "2060"
  assert precompiler.locals["scenario"] == "baseline"
  assert precompiler.locals["next_year"] == "2060"


def test_import_file_is_parsed_with_existing_state(tmp_path):
  imported = tmp_path / "imported.gms"
  imported.write_text("""
  $GROUP G_imported
    q[t] "Imported quantity";
  """, encoding="utf-8")
  text = """
  $Import imported.gms
  $LOOP G_imported:
    imported_{name}{sets} = {name}.L{sets};
  $ENDLOOP
  """

  output, precompiler = expand(tmp_path, text)

  assert "Import file:" in output
  assert "Variable q[t] \"Imported quantity\" //;" in output
  assert_contains_line(output, "imported_q[t] = q.L[t];")
  assert "q" in precompiler.groups["G_imported"]


def test_if_statement_supports_gams_style_comparisons(tmp_path):
  text = """
  $SET scenario baseline
  $IF "%scenario%" EQ "baseline":
    scalar active /1/;
  $ENDIF
  $IF 1 NE 1:
    scalar inactive /1/;
  $ENDIF
  """

  output, _ = expand(tmp_path, text)

  assert "scalar active /1/;" in output
  assert "scalar inactive /1/;" not in output
  assert "# If condition evaluated to false" in output


def test_for_loop_supports_single_and_tuple_iterators(tmp_path):
  text = """
  $FOR i in range(2):
    scalar n_i /i/;
  $ENDFOR
  $FOR {name}, {value} in [("a", 1), ("b", 2)]:
    scalar {name} /{value}/;
  $ENDFOR
  """

  output, _ = expand(tmp_path, text)

  assert_contains_line(output, "scalar n_0 /0/;")
  assert_contains_line(output, "scalar n_1 /1/;")
  assert_contains_line(output, "scalar a /1/;")
  assert_contains_line(output, "scalar b /2/;")


def test_user_defined_function_replaces_arguments(tmp_path):
  text = """
  $FUNCTION init(name, sets):
    name.Lsets = 1;
  $ENDFUNCTION
  @init(q, [t])
  """

  output, precompiler = expand(tmp_path, text)

  assert "Define function: init" in output
  assert "q.L[t] = 1;" in output
  assert "init" in precompiler.user_functions


def test_replace_and_regex_macros_support_counts_and_groups(tmp_path):
  text = """
  $REPLACE("foo", "bar", 1)
    foo foo
  $ENDREPLACE
  $REGEX("(x)([0-9])", "\\g<1>_\\g<2>")
    x1 x2
  $ENDREGEX
  """

  output, _ = expand(tmp_path, text)

  assert "bar foo" in output
  assert "x_1 x_2" in output


def test_block_equation_dollar_condition_without_parentheses(tmp_path):
  text = """
  $BLOCK B_market
    E_q[t]$tx0[t].. q[t] =E= demand[t];
  $ENDBLOCK
  """

  output, precompiler = expand(tmp_path, text)

  assert "E_q[t]$(tx0[t]).." in output
  assert precompiler.blocks["B_market"]["E_q"].conditions == "$(tx0[t])"


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


def test_model_define_combines_and_removes_equations(tmp_path):
  text = """
  $BLOCK B_core
    E_q[t].. q[t] =E= demand[t];
    E_p[t].. p[t] =E= 1;
  $ENDBLOCK
  $MODEL M_subset
    B_core
    -E_p
  ;
  """

  output, precompiler = expand(tmp_path, text)

  assert "MODEL M_subset /" in output
  assert "E_q" in precompiler.blocks["M_subset"]
  assert "E_p" not in precompiler.blocks["M_subset"]
  assert re.search(r"MODEL M_subset / \s*E_q\s*/;", output)


def test_loop_over_equations_exposes_name_sets_conditions_lhs_and_rhs(tmp_path):
  text = """
  $BLOCK B_market
    E_q[t]$(tx0[t]).. q[t] =E= demand[t];
  $ENDBLOCK
  $LOOP B_market:
    copy_{name}{sets}${conditions}.. {LHS} =E= {RHS};
  $ENDLOOP
  """

  output, precompiler = expand(tmp_path, text)

  assert "copy_E_q[t]$[((tx0[t]))]..  q[t]  =E=  demand[t];" in output
  assert precompiler.blocks["B_market"]["E_q"].conditions == "$((tx0[t]))"


def test_fix_and_unfix_expand_groups_with_conditions_and_bounds(tmp_path):
  text = """
  $GROUP G_endo
    q[t]$(tx0[t]) "Quantity";

  $FIX G_endo;
  $FIX(0) q[t];
  $UNFIX(0, inf) G_endo;
  """

  output, _ = expand(tmp_path, text)

  assert "q.FX[t]$((tx0[t])) = q.L[t];" in output
  assert "q.FX[t] = 0;" in output
  assert "q.lo[t]$((tx0[t])) = 0;" in output
  assert "q.up[t]$((tx0[t])) = inf;" in output


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


def test_save_and_read_preserve_precompiler_metadata(tmp_path):
  first_path = tmp_path / "first.gms"
  first_path.write_text("""
  $GROUP G_endo
    q[t] "Quantity";
  """, encoding="utf-8")
  first = gamy.Precompiler(first_path)
  first()
  first.save("checkpoint")

  second_path = tmp_path / "second.gms"
  second_path.write_text("""
  $LOOP G_endo:
    restored_{name}{sets} = {name}.L{sets};
  $ENDLOOP
  """, encoding="utf-8")
  second = gamy.Precompiler(second_path)
  second.read("checkpoint")
  output = second()

  assert_contains_line(output, "restored_q[t] = q.L[t];")
  assert "q" in second.groups["G_endo"]


def test_find_gams_reports_missing_executable(monkeypatch):
  monkeypatch.setattr(gamy.shutil, "which", lambda name: None)
  monkeypatch.delenv("GAMS", raising=False)
  monkeypatch.delenv("gams", raising=False)

  with pytest.raises(SystemExit, match="could not find GAMS"):
    gamy.find_gams()


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
