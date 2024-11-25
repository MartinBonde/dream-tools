# -*- coding: utf-8 -*-
"""
A preprocessor for GAMS files (.gms files), implementing a number of additional features.
Basic syntax:
  Nothing is case sensitive, just like in GAMS
  All commands (preprocessor macros) start with a "$"
  '#'' is used for comments instead of '*'
  Items can be separated by commas, line breaks, or a mix.
  Commands are processed in the order they appear and can be nested.
  Flow control statements (IF, LOOP, FOR, and FUNCTION) are processed before other commands.
  Nested flow control commands must be identified with a number, e.g.
    $IF1 <condition>:
      $IF2 <condition>:
      $ENDIF2
    $ENDIF1
Available commands:
  $GROUP <variables or groups>;
    A group is a data structure containing variables.
    New variables should always be defined using the $GROUP command rather than a GAMS VARIABLES statement.
    The group command groups together variables so that they can be manipulated together more easily, using other gamY commands such as $FIX, $UNFIX, $LOOP, $DISPLAY, or $GROUP.
    Variable elements can be selectively included in a group using dollar conditions. Note that conditions ALWAYS need to be enclosed in round brackets.
    Groups can be added together (union operation) or removed (complement operation).
    Example:
      $GROUP G_newGroup
        var1[t]   "label for variable 1"
        var2[a,t]$(a.val >= 18) "label for variable 2"
        G_oldGroup
        var3[a,t], -var3$(a.val < 18)  # Equivalent to var2
      ;

  $BLOCK <block name> [$condition] <equations> $ENDBLOCK
    A block is a data structure containing equations.
    New equations should always be defined using the $BLOCK command rather than a GAMS EQUATIONS statement.
    The block command bundles together equations so that they can be manipulated together more easily, using other gamY commands such as $LOOP or $MODEL.
    Example:
      $BLOCK B_myBlock [$conditon]
        E_eq1[t].. v1[t] =E= v2;
      $EndBlock

    Instead of manually providing a unique equation identifier, the user may instead explicitly map the equation to an endogenous variable.
    A group of variables mapped to the block equations is created with the group name provided (e.g. myBlock_endogenous below).
    Example:
      $BLOCK myBlock_equations myBlock_endogenous $(t.val > t1.val)
        v1[t]$(tEnd[t]).. v1[t] =E= v2;
      $EndBlock  
      $FIX All; $UNFIX myBlock_endogenous;
      solve myBlock_model using CNS;

  $MODEL <model name> <equations and/or blocks/models>;
    $MODEL can use a mix of Models, Blocks, and Equations.
    Example:
      $MODEL M_newModel
        M_oldModel
        B_myBlock
        E_eq2
      ;

  $LOOP <group name>: <content replacing {NAME}, {SETS}, {CONDITIONS}> $ENDLOOP
    Example:
      $LOOP G_myGroup:
        parameter saved_{name}{sets};
        saved_{name}{sets} = {name}.L{sets};
      $EndLoop

  $LOOP <block name>: <content replacing {NAME}, {SETS}, {CONDITIONS}, {LHS}, or {RHS}> $ENDLOOP
    Example:
      $LOOP B_myBlock:
        {name}_ss{sets}$(tx0[t] and {conditions}) {LHS} =E= {RHS};
      $EndLoop

  $FIX <variables or groups>;
    Exogenous variables.
    Equivalent to writing
    <variable>.fx[<variable>] = <variable>.l[<sets>];
    for each variable.

  $UNFIX[(<lower bound>, <upper bound>)] <variables or groups>;
    Endogenize variables.
    Equivalent to writing
    <variable>.lo[<variable>] = <lower bound>;
    <variable>.up[<variable>] = <upper bound>;
    for each variable.
    If no bounds are given, lower and upper bounds are set to -inf and inf respectively.

  $DISPLAY <variables or groups>;

  $IMPORT <filename>
    Include a separate file in this file. Note that the regular GAMS $INCLUDE still exists (and is faster). Use @IMPORT if imported file should be processed with gamY.

  $IF <condition expression>: <content> $ENDIF
    If statement where the condition is evaluated in python.

  $FOR <python expression>: <content> $ENDFOR
    $For {parameter}, {value} in [("a", 1), ("b",2)]:
      {parameter} = {value};
    $EndFor

  $FUNCTION <function name>([<argument name>, <...>]) $ENDFUNCTION
    Define a new gamY function.
  @<function name>([<argument value>, <...>])
    Call a previously defined function.

  $REPLACE $ENDREPLACE
    Find and replace.

  $REGEX $ENDREGEX
    Find and replace with regular expressions (wild card search).
    
Save/Read options
  As in regular GAMS, runs can be saved and read using the options s=<filename> and r=<filename>
  Group definitions etc. are saved in and read from a pkl file along side the GAMS g00 file.

Implementation notes
  The preprocessor uses 'brute force' regular expressions to find and replace the new macro commands.
  It is recommended to use an AI to understand or modify any regular expression patterns.
  Commands are processed in the order they appear and can be nested (using recursive descent parsing).
  A tokenizer is not used and is unlikely to simplify the code as syntax varies for each command,
  and only the commands, rather than all the GAMS code, need to be parsed.
  The program can be rewritten using a tokenizer, lexer, and parser to allow for unlimited nesting of flow control statements,
  and to make the syntax more robust.
"""
import sys
import os
import shutil
import subprocess
import pickle
import re
from math import ceil, floor
from heapq import heappush
from timeit import default_timer as timer
from textwrap import dedent

import itertools # Useful in FOR loops in MAKRO

# gamY data objects
from .classes import Variable, Equation, Function, MockMatch, Group, Block, CaseInsensitiveDict

#  The regex patterns used to match commands
from .patterns import PATTERNS, open_bracket, close_bracket, brackets, no_brackets, ident

# global gamY settings
leave_env_variables_for_gams = False
automatic_additive_residuals_prefix = None
automatic_multiplicative_residuals_prefix = None
error_on_missing_label = True
variable_equation_prefix = ""
automatic_dummy_suffix = None # If set, a dummy variable is created for each variable in a group with the suffix

def get_lst_path(file_path):
  """Return the path to the LST file corresponding to the GAMS file"""
  file_dir, file_name = os.path.split(file_path)
  list_file_dir = os.path.join(file_dir, "LST")
  if not os.path.exists(list_file_dir):
    os.makedirs(list_file_dir)
  list_file_name = file_name.replace(".gms", ".lst")
  return os.path.join(list_file_dir, list_file_name)


class Precompiler:
  """Object with methods to parse each gamY command"""

  def __init__(self, file_path, patterns=PATTERNS, additive_adjust=None, multiplicative_adjust=None):
    self.groups = CaseInsensitiveDict({"all": {}})
    self.groups_conditions = CaseInsensitiveDict({"all": {}})
    self.par_groups = CaseInsensitiveDict({"all": {}})
    self.par_groups_conditions = CaseInsensitiveDict({"all": {}})
    self.set_groups = CaseInsensitiveDict({"all": {}})
    self.set_groups_conditions = CaseInsensitiveDict({"all": {}})
    self.blocks = CaseInsensitiveDict()
    self.globals = dict(os.environ)
    self.user_functions = CaseInsensitiveDict()
    self.locals = {}

    self.has_read_file = False

    self.adjustment_terms = []
    if additive_adjust is not None:
      self.additive_adjust = additive_adjust
    else:
      self.additive_adjust = automatic_additive_residuals_prefix
    if multiplicative_adjust is not None:
      self.multiplicative_adjust = multiplicative_adjust
    else:
      self.multiplicative_adjust = automatic_multiplicative_residuals_prefix
    
    if additive_adjust:
      self.adjustment_terms.append(additive_adjust)
    if multiplicative_adjust:
      self.adjustment_terms.append(multiplicative_adjust)
    for term in self.adjustment_terms:
      self.groups[term] = Group()
      self.groups_conditions[term] = {}

    self.model_counter = 0  # As models cannot be redefined, we increment the names of temporary models defined

    self.patterns = patterns
    self.file_path = os.path.abspath(file_path)
    self.file_dir, self.file_name = os.path.split(self.file_path)

    self.list_file_path = get_lst_path(self.file_path)

    #  In cases where a command should be parsed by gamY, but also left for GAMS to parse, we replace special characters temporarily
    self.DOLLAR_SUB = "¤dollar¤"
    self.SEMICOLON_SUB = "¤semicolon¤"
    self.PERCENT_SUB = "¤percent¤"
    self.AT_SUB = "¤at¤"
    self.star_comment_pattern = re.compile(r"^\*.*", re.MULTILINE)
    self.hashtag_comment_pattern = re.compile(r"\#.*", re.MULTILINE)
    self.block_comment_pattern = re.compile(r"^\$ontext.*?\$offtext", re.IGNORECASE | re.MULTILINE | re.DOTALL)

    #  Maps between regex patterns and the correpsonding method for each gamY command
    self.recursive_commands = [
      ("set", self.set_env_variable),
      ("user_function", self.user_function),
      ("eval", self.eval),
      ("import", self.import_file),
      ("block", self.block_define),
      ("solve", self.solve),
      ("model", self.model_define),
      ("group", self.group_define),
      ("par_group", self.par_group_define),
      ("set_group", self.set_group_define),
      ("replace", self.sub),
      ("regex", self.regex),
      ("display", self.display),
      ("display_all", self.display_all),
      ("fix", self.fix_unfix),
    ]
    self.top_down_commands = [  # Remeber also to add these to the top down pattern in patterns.py
      ("define_function", self.define_function),
      ("env_variable", self.insert_env_variable),
      ("if", self.if_statements),
      ("for_loop", self.for_loop),
      ("loop", self.loop),
      ("eval_python", self.eval_python),
    ]
    self.commands = self.recursive_commands + self.top_down_commands
    self.prev_match = None

  @property
  def equations(self):
    return {k: v for block in self.blocks.values() for k, v in block.items()}

  def warning(self, msg):
    print("WARNING: " + msg)
    return "**** " + msg

  def log(self, msg):
    # print(msg)
    return msg

  def error(self, msg):
    error_text = "ERROR!\ngamY could not process the file due to the following error:\n" + msg
    with open(self.list_file_path, 'w+') as file:
      file.write(error_text)
    sys.exit(error_text)

  def __call__(self):
    """Read GAMS file"""
    with open(self.file_path, 'r') as f:
      text = "\n" + f.read()
    if not self.has_read_file:
      text = "$ONEOLCOM\n$EOLCOM #\n\n" + text  # Add option for # comments if this is the first file being run
    self.processed_text = ""
    text = self.parse(text, top_level=True)
    text = self.processed_text + text
    text = self.dedent_dollar(text)
    text = self.restore_temporary_substitutions(text)
    text = self.remove_unnecessary_on_off_listing(text)
    return text
  
  @staticmethod
  def remove_unnecessary_on_off_listing(text):
    pattern = re.compile(r"\$offlisting\s*\$onlisting\s?", re.MULTILINE | re.IGNORECASE)
    return pattern.sub("", text)

  def parse(self, text, top_level=False):
    while True:
      text = self.clean_comments(text)
      match = self.patterns["Any"].search(text)
      if match and (repr(match) == repr(self.prev_match) and text == self.prev_text):
        match = self.patterns["Any"].search(text, match.start() + 1)  # Check one character in, so that we don't match the same pattern forever
      self.prev_match = match
      self.prev_text = text
      if top_level and match:
        self.processed_text += text[:match.start()]
        text = text[match.start():]
      if match:
        # Check if commands needs top parsed right away, instead of recursively
        if self.patterns["TopDown"].fullmatch(match.group(0)):
          # print(f"TopDown approach used for match: {match.group(0)[:10]} [..] {match.group(0)[-10:]}")
          text = text.replace(match.group(0), self.process_command(match.group(0)), 1)
        else:
          text = text.replace(match.group(0), self.parse(match.group(0)), 1)  # Recursively parse
      else:
        return self.process_command(text)  # Process commands when inner scope of recursion is reached

  def round_parentheses(self, text):
    return text.replace("[", "(").replace("]", ")")

  def process_command(self, text):
    for command, func in self.commands:
      match = self.patterns[command].fullmatch(text)  # match method is used (rather than search) to avoid matching inside commands when parsing top down
      if match:
        #  print("***MATCH***", command)
        text = text.replace(match.group(0), func(match, text), 1)
    return text

  def read(self, file_name):
    """
    Read blocks, groups, and variables from saved file if the r=<file_name> option is used.
    """
    self.has_read_file = True
    try:
      with open(os.path.join(self.file_dir, file_name) + ".pkl", 'rb') as f:
        (self.blocks,
         self.groups, self.groups_conditions,
         self.par_groups, self.par_groups_conditions,
         self.set_groups, self.set_groups_conditions,
         loaded_globals, self.user_functions,
        ) = pickle.load(f)
        print("Precompiler file read: " + file_name + ".pkl")
      self.globals.update(loaded_globals)
    except FileNotFoundError:
      self.warning(f"gamY read file not found ({file_name}.pkl), macro groups have been reset")

  def save(self, file_name):
    """
    Save dicts of blocks, groups, and variables in pickle file if s=<file_name> option is used
    """
    with open(os.path.join(self.file_dir, file_name) + ".pkl", 'wb') as f:
      pickle.dump((
          self.blocks, 
          self.groups, self.groups_conditions,
          self.par_groups, self.par_groups_conditions,
          self.set_groups, self.set_groups_conditions,
          self.globals, self.user_functions
        ),
        f, pickle.HIGHEST_PROTOCOL
      )

  def comment_out(self, text):
    return_string = text.replace("\n", "\n#")
    return self.clean_comments(return_string)

  def clean_comments(self, text):
    """
    Return string with special characters removed from comments.
    """
    for pattern in (self.star_comment_pattern, self.hashtag_comment_pattern, self.block_comment_pattern):
      for match_text in pattern.findall(text):
        replacement_text = match_text
        replacement_text = replacement_text.replace("$", self.DOLLAR_SUB)
        replacement_text = replacement_text.replace(";", self.SEMICOLON_SUB)
        replacement_text = replacement_text.replace("%", self.PERCENT_SUB)
        replacement_text = replacement_text.replace("@", self.AT_SUB)
        text = text.replace(match_text, replacement_text)
    return text

  def restore_temporary_substitutions(self, text):
    """
    Return string with special characters reinserted.
    """
    text = text.replace(self.DOLLAR_SUB, "$")
    text = text.replace(self.SEMICOLON_SUB, ";")
    text = text.replace(self.PERCENT_SUB, "%")
    text = text.replace(self.AT_SUB, "@")
    return text

  @staticmethod
  def remove_comments(text):
    """
    Return string with comments removed.
    """
    star_comment_pattern = re.compile(r"^\#.*", re.MULTILINE)
    hashtag_comment_pattern = re.compile(r"#.*", re.MULTILINE)
    block_comment_pattern = re.compile(r"^\$ontext.*?\$offtext", re.IGNORECASE | re.MULTILINE | re.DOTALL)

    for pattern in (star_comment_pattern, hashtag_comment_pattern, block_comment_pattern):
      text = pattern.sub("", text)
    return text

  def insert_env_variable(self, match, text):
    """
    Replace environmental variables with their value, e.g. %variable_name%
    """
    key = match.group(1)

    in_scope = self.in_scope()
    if key in in_scope:
      return in_scope[key]
    else:
      self.warning(f"\\%{key}\\% is not defined and was not replaced")
      return self.PERCENT_SUB + key + self.PERCENT_SUB

  def user_function(self, match, text):
    """
    Insert user defined function call, e.g. @my_funct(), defined using $FUNCTION
    """
    func_name, args = match.group(1), match.group(2)
    if func_name not in self.user_functions:
      self.error(f"@{func_name} has not been defined: {match.group(0)}")

    func = self.user_functions[func_name]
    args = re.findall(r"(?:[^,\[]+|(?:\[[^\]]*\]))+", args)
    replacement_text = func.expression
    self.log(f"User function <{func_name}> called with {args}")
    for arg_name, arg in zip(func.args, args):
      replacement_text = replacement_text.replace(arg_name, arg.strip())
    return replacement_text

  def in_scope(self):
    in_scope = {}
    in_scope.update(self.globals)
    in_scope.update(self.locals)  # locals overwrite globals
    in_scope = {k: v for k, v in in_scope.items()}
    return in_scope

  def set_env_variable(self, match, text, eval_command=False, leave_command_for_GAMS=None):
    """
    Read $set, $setglobal, and $setlocal commands
    The commands are left intact to also be processed by GAMS
    """
    key = match.group(2)
    val = match.group(3)
    if eval_command:
      val = str(eval(val))
    if match.group(1).lower() == "global":
      self.globals[key] = val
    else:
      self.locals[key] = val
    if leave_command_for_GAMS is None:
      leave_command_for_GAMS = leave_env_variables_for_gams
    if leave_command_for_GAMS:
      replacement_text = match.group(0).replace("$", self.DOLLAR_SUB).lstrip()
    else:
      replacement_text = ""
    return replacement_text

  def eval(self, match, text):
    return self.set_env_variable(match, text, eval_command=True)

  def if_statements(self, match, text):
    """
    Parse $If .. $EndIf command.
    To nest if statements use an id, e.g. $IF1 $ENDIF1
    """
    id_ = match.group(1)
    condition = self.parse(match.group(2))
    expression = match.group(3)

    if not id_:
        id_ = " "
    if f"$IF{id_}" in expression:
      self.error(
        f"""
        Nested IF statements must be identified with id numbers (e.g. $IF1 .. $ENDIF1).
        Condition: '{condition}'
        Expression: '{expression}'
        """
      )

    condition = re.sub(r"(?<![=><!])=(?![=><!])", "==", condition)
    condition = re.sub(r"<>", "!=", condition)
    condition = re.sub(r"(?<![a-zA-Z])EQ(?![a-zA-Z])", "==", condition, flags=re.IGNORECASE)
    condition = re.sub(r"(?<![a-zA-Z])NE(?![a-zA-Z])", "!=", condition, flags=re.IGNORECASE)
    condition = re.sub(r"(?<![a-zA-Z])LT(?![a-zA-Z])", "<", condition, flags=re.IGNORECASE)
    condition = re.sub(r"(?<![a-zA-Z])GT(?![a-zA-Z])", ">", condition, flags=re.IGNORECASE)
    condition = re.sub(r"(?<![a-zA-Z])LE(?![a-zA-Z])", "<=", condition, flags=re.IGNORECASE)
    condition = re.sub(r"(?<![a-zA-Z])GE(?![a-zA-Z])", ">=", condition, flags=re.IGNORECASE)

    condition_trunc = condition.split("\n")[0]
    replacement_text = f"\n# ----- gamY: IF {condition_trunc}: -----\n"

    try:
      if eval(condition.lower()):
        replacement_text += expression
      else:
        replacement_text += "# If condition evaluated to false"
    except Exception as e:
      self.error(f"""Failed to evaluate IF-condition: {condition}
{e}""")
    replacement_text += f"\n# ----- gamY: ENDIF -----\n"

    return replacement_text

  def import_file(self, match, text):
    """
    Return text with $Import commands replaced by code in Import file
    """
    file_name = os.path.join(self.file_dir, match.group(1))
    replacement_text = f"\n# ----- gamY: Import file: {file_name} ----- \n"
    if os.path.isfile(file_name):
      try:
        with open(file_name, "r") as f:
          replacement_text += self.parse("\n"+f.read())
      except FileNotFoundError:
        replacement_text += self.warning(f"File was found, but could not be read: '{file_name}'")
    else:
      replacement_text += self.warning(f"File not found: '{file_name}'")

    return replacement_text

  def sub(self, match, text):
    """
    Parse $REPLACE command
    Example:
      $REPLACE('t', 'tt'):
        t_in_name[t]
      $ENDREPLACE
      ->
      tt_in_name(tt)
    """
    old, new, count, expression = match.groups()
    if count:
      return expression.replace(old[1:-1], new[1:-1], int(count[1:]))
    else:
      return expression.replace(old[1:-1], new[1:-1])

  def regex(self, match, text):
    """
    Parse $REGEX command
    First argument should be a string or python code without commas.
    Second argument must be a string enclosed in '' or ""
    Third arguments is an optional integer
    Example:
      $REGEX('\bt\b', 'tt'):
        t_in_name[t]
      $ENDREGEX
      ->
      t_in_name(tt)
    """

    id_, old, new, count, expression = match.groups()

    if not id_:
        id_ = " "
    if f"$REGEX{id_}" in expression:
      self.error(
        f"""
        Nested REGEX statements must be identified with id numbers (e.g. $REGEX1 .. $ENDREGEX1).
        Expression: '{expression}'
        """
      )

    # If argument is not a string, evaluate it
    if not (old[0] == "'" or old[0] == '"'):
      old = eval(old)
    else:
      old = old[1:-1]

    pattern = re.compile(old, re.IGNORECASE | re.MULTILINE)
    if count:
      return pattern.sub(new[1:-1], expression, int(count[1:]))
    else:
      return pattern.sub(new[1:-1], expression)

  # def generate_equation_name(self, var, sets, conditions):
  #   """Generate unique equation name based on the name, sets, and conditions of the associated endogenous variable."""
  #   suffix = self.sets_to_conditions(sets, var, conditions)
  #   if not suffix:
  #     return f"{variable_equation_prefix}{var.name}"

  #   # Remove sets denoted with square brackets, special characters, and whitespace
  #   suffix = re.sub(r"\[.+?\]", "", suffix)
    
  #   # Replace comparison operators with their corresponding abbreviations
  #   suffix = re.sub(r">=", "GE", suffix)
  #   suffix = re.sub(r"<=", "LE", suffix)
  #   suffix = re.sub(r"<", "LT", suffix)
  #   suffix = re.sub(r">", "GT", suffix)
  #   suffix = re.sub(r"=", "EQ", suffix)
  #   suffix = re.sub(r"<>", "NE", suffix)

  #   # Replace whitespace with underscores
  #   suffix = re.sub(r"\s+", "_", suffix)

  #   # Remove other special characters
  #   suffix = re.sub(r"[^A-Za-z0-9_]", "", suffix)

  #   return f"{variable_equation_prefix}{var.name}_{suffix}"

  def generate_equation_name(self, name, sets, suffix):
      if sets and not suffix:
        suffix = "_" + sets[1:-1].replace(",", "_").replace(" ", "")
        if suffix.endswith("_t"):
          suffix = suffix[:-2]
        if name.endswith(suffix):
          suffix = ""
      return variable_equation_prefix + name + suffix

  @staticmethod
  def merge_conditions(*args):
    args = [arg.strip("$") for arg in args if arg]
    if args:
      return f"$({' and '.join(args)})"
    return ""

  def check_if_variable_exists(self, name):
    if name not in self.groups["all"]:
      self.error(f"Variable '{name}' must be defined before being used in a block.")

  def create_additive_adjustment_term(self, model_name, name, sets, RHS, replacement_text):
    j_name = self.additive_adjust + name
    j_group_name = f"{self.additive_adjust}_{model_name}"
    docstring = f"Additive adjustment term for equation {name}"
    replacement_text += f" VARIABLE {j_name}{sets} \"{docstring}\"; {j_name}.FX{sets} = 0;"
    if j_group_name not in self.groups:
      self.groups[j_group_name] = Group()
      self.groups_conditions[j_group_name] = {}
    j_var = Variable(j_name, sets, docstring)
    self.groups[self.additive_adjust][j_name] = j_var
    self.groups["all"][j_name] = j_var
    self.groups[j_group_name][j_name] = j_var
    if RHS.strip()[0] == "-":
      RHS = f"{self.additive_adjust}{name}{sets} {RHS}"
    else:
      RHS = f"{self.additive_adjust}{name}{sets} + {RHS}"
    return RHS, replacement_text

  def create_multiplicative_adjustment_term(self, model_name, name, sets, RHS, replacement_text):
    j_name = self.multiplicative_adjust + name
    j_group_name = f"{self.multiplicative_adjust}_{model_name}"
    docstring = f"Multiplicative adjustment term for equation {name}"
    replacement_text += f"VARIABLE {j_name}{sets} \"{docstring}\"; {j_name}.FX{sets} = 0;"
    if j_group_name not in self.groups:
      self.groups[j_group_name] = Group()
      self.groups_conditions[j_group_name] = {}
    j_var = Variable(j_name, sets, docstring)
    self.groups[self.multiplicative_adjust][j_name] = j_var
    self.groups["all"][j_name] = j_var
    self.groups[j_group_name][j_name] = j_var
    RHS = f"(1+{self.multiplicative_adjust}{name}{sets}) * ({RHS})"
    return RHS, replacement_text

  def block_define(self, match, text):
    """
    Block command syntax example:
    $Block B_block_name
      E_eq1[t].. v1[t] =E= v2;
    $EndBlock
    ==>
    #  ***block_name****
    Equation E_eq1[t];
    Variable j_eq1[t];
    j_eq1.L[t] = 0;
    Variable jr_eq1[t];
    jr_eq1.L[t] = 0;
    E_eq1[t].. v1[t] =E= (1+jr_eq1[t]) * (j_eq1[t] + v2);
    """
    equation_pattern = re.compile(fr"""
      (?:^|\,)             #  Check only beginning of line or after a comma.
      \s*                  #  Ignore whitespace
      ({ident})?(\&{ident})?         #  Name of equation, Suffix
      
      ({open_bracket}[^$]+?{close_bracket})?        #  Sets
      \s*
      (\$.+?)?              #  Set restrictions
      \s*
      \.\.
      (.+?)                 #  LHS
      =E=
      (.+?)\;              #  RHS
    """, re.VERBOSE | re.MULTILINE | re.DOTALL | re.IGNORECASE)

    lhs_variable_pattern = re.compile(f"({ident})({open_bracket}[^$]+?{close_bracket})?", re.IGNORECASE | re.MULTILINE)

    model_name, group_name, block_conditions, content = match.groups()
    replacement_text = f"\n# ----- gamY: Initialize {model_name} equation block -----\n"
    self.blocks[model_name] = Block()
    replacement_text += f"\n$GROUP {group_name} ;\n"
    for equation_match in equation_pattern.finditer(self.remove_comments(content)):
      name, suffix, sets, conditions, LHS, RHS = (
        group if group is not None else "" for group in equation_match.groups()
      )

      if not name:
        name, var_sets = lhs_variable_pattern.search(LHS).groups()
        if not sets:
          sets = var_sets

      if group_name:
        self.check_if_variable_exists(name)

      merged_conditions = self.merge_conditions(block_conditions, conditions)

      if name in self.groups["all"] and automatic_dummy_suffix:
        merged_conditions = self.merge_conditions(merged_conditions, f"{name}{automatic_dummy_suffix}{sets}")
      
      if group_name:
        replacement_text += f"$GROUP+ {group_name} {name}{sets}{merged_conditions};"
    
      if self.additive_adjust:
        RHS, replacement_text = self.create_additive_adjustment_term(model_name, name, sets, RHS, replacement_text)
      if self.multiplicative_adjust:
        RHS, replacement_text = self.create_multiplicative_adjustment_term(model_name, name, sets, RHS, replacement_text)

      eq_name = self.generate_equation_name(name, sets, suffix[1:]) if group_name else name

      eq = Equation(eq_name, sets, merged_conditions, LHS, RHS)
      self.blocks[model_name][eq.name] = eq
      replacement_text += f"EQUATION {eq.name}{eq.sets};"
      replacement_text += "\n"+f"{eq.name}{eq.sets}{eq.conditions}.. {eq.LHS} =E= {eq.RHS};"+"\n"

    replacement_text += f"$MODEL {model_name} {model_name};"
    return replacement_text

  def model_define(self, match, text):
    """
    Parse $MODEL command.
    Define models from blocks, models, and equations.
    Syntax example:
    $MODEL M_myNewModel
      M_oldModel
      B_myBlock
      E_eq
    ;
    """
    item_pattern = re.compile(fr"""
      (?:^|\,)             #  Check only beginning of line or after a comma.
      \s*                  #  Ignore whitespace
      (\-)?                #  Optional MINUS character if block or equation is to be removed instead of added ($1)
      ({ident})            #  Name of equation ($2)
    """, re.VERBOSE | re.MULTILINE | re.IGNORECASE)

    equations = CaseInsensitiveDict({eq.name: eq for block in self.blocks.values() for eq in block.values()})
    model_name = match.group(1)
    content = self.remove_comments(match.group(2))
    replacement_text = f"\n# ----- gamY: Initialize {model_name} model -----\n"
    replacement_text += f"MODEL {model_name}"
    new_model = Block()
    for item_match in item_pattern.finditer(content):
      remove = item_match.group(1)
      name = item_match.group(2)
      if name in self.blocks:
        for eq in self.blocks[name].values():
          if remove:
            new_model.pop(eq.name)
          else:
            new_model[eq.name] = eq
      elif name in equations:
        if remove:
          if name in new_model:
            new_model.pop(name)
          else:
            self.warning(
              f"Equation {name} could not be removed from {model_name}, as it is not part of that block or model.")
        else:
          new_model[name] = equations[name]
      else:
        raise KeyError(name, " is not a valid equation, block of equations, or model")

    if new_model: # If the model is not empty
      replacement_text += " / \n"
      replacement_text += ", ".join([eq.name for eq in new_model.values()])
      replacement_text += "\n /"
    replacement_text += ";"

    #  Define an equation block, so that the model can be used in the same ways a regular blocks
    self.blocks[model_name] = new_model

    #  Define a group of all the adjustment variables in model, with the name "Adjust_[model_name]"
    for j in self.adjustment_terms:
      j_group_name = f"{j}_{model_name}"
      if j_group_name not in self.groups:
        self.groups[j_group_name] = {}
        self.groups_conditions[j_group_name] = {}
        for eq in new_model.values():
          j_name = j+eq.name
          self.groups[j_group_name][j_name] = self.groups[j][j_name]

    return replacement_text

  def sets_to_conditions(self, set, var, item_conditions):
    """
    Used in Group defition to turn sub-sets into dollar-conditons
    x[a,b]
    x[a1,'b']$(a2[a1]) -> x[a,b]$(a1[a] and sameas['b',b] and a2[a])
    """
    condition_sets = set[1:-1].split(",")
    definition_sets = var.sets[1:-1].split(",")
    for con_set, def_set in zip(condition_sets, definition_sets):
      if con_set == def_set:
        continue
      if item_conditions:
        item_conditions = re.sub(fr"\b{con_set}\b", def_set, item_conditions, flags=re.IGNORECASE)
      if "'" in con_set or '"' in con_set:
        item_conditions = self.combine_conditions(item_conditions, f"sameas({con_set},{def_set})")
      else:
        item_conditions = self.combine_conditions(item_conditions, f"{con_set}[{def_set}]")
    return item_conditions

  def group_define(self, match, text, sym_type="Variable"):
    """
    Parse $GROUP command
    Syntax example:
    $GROUP G_newGroup
      var1[a,t] "label for variable 1"
      var2[t]   "label for variable 2"
      var3, var4
      G_oldGroup
    ;
    """
    group_variable_pattern = re.compile(fr"""
      (?:^|\,)              #  Check only beginning of line or after a comma.
      \s*                   #  Ignore whitespace
      (\-)?                 #  Optional MINUS character, if group or variable should be removed rather than added ($1)
      \s*
      ({ident})             #  Name of variable ($2)
      ({open_bracket}{no_brackets}+?{close_bracket})? #  Optional sets  ($3)
      (\${open_bracket}.+?{close_bracket})?           #  Optional conditions  ($4)
      \s*
      ('.*?'|".*?")?        #  Optional label  ($5)
      (\s+-?\d+(?:\.\d*)?)? #  Optional initial value of variable  ($6)
      \s*
      (?=[\n\,\;])          #  Variable separator (comma or new line)
    """, re.VERBOSE | re.MULTILINE | re.IGNORECASE)

    if sym_type == "Parameter":
      GROUPS = self.par_groups
      CONDITIONS = self.par_groups_conditions
      L = ""
      DUMMY_SUFFIX = None
    elif sym_type == "Set":
      GROUPS = self.set_groups
      CONDITIONS = self.set_groups_conditions
      L = ""
      DUMMY_SUFFIX = None
    else:
      GROUPS = self.groups
      CONDITIONS = self.groups_conditions
      L = ".L"
      DUMMY_SUFFIX = automatic_dummy_suffix

    add_to_existing, group_name, content = match.groups()
    content = self.remove_comments(content)
    if add_to_existing:
      content  = group_name + ", " + content
      replacement_text = "$offlisting\n"
    else:
      replacement_text = f"# ----- gamY: Initialize {group_name} group -----\n$offlisting\n"

    new_group = Group()
    new_group_conditions = CaseInsensitiveDict()

    #  Loop over variables and groups to be added or removed from group
    for item in group_variable_pattern.finditer(content):
      remove, name, sets, item_conditions, label, level = item.group(1, 2, 3, 4, 5, 6)

      if name in GROUPS:
        symbols = GROUPS[name].values()
        old_group_conditions = CONDITIONS[name]
      elif name in GROUPS["all"]:
        symbols = (GROUPS["all"][name],)
        old_group_conditions = {}
      elif remove:
        self.error(
          f"{name} is not a {sym_type} or group, and could not be removed from {group_name} (this might be due to a typo).")
      else:  # Initialize a new variable in a dummy group to loop over.
        symbols = (Variable(name, sets, label),) # Variable class is also used for sets and parameters
        old_group_conditions = {}
        if not label:
          f = self.error if error_on_missing_label else self.warning
          f(f"{name} was defined without an explanatory text in {sym_type} group {group_name} (this might be due to a typo).\n{match.groups()}")

      for var in symbols:
        if sets and "$" in sets:
          self.error(
            f"""Conditionals in the GROUP statement must be surrounded by parentheses, e.g. $(d1[d]).
Error in {group_name}: {name}{sets}{item_conditions}""")

        if sets and (sets != var.sets):
          item_conditions = self.sets_to_conditions(sets, var, item_conditions)

        if remove:
          if var.name not in new_group:
            replacement_text += f"# {var.name} is not in group and could therefor not be removed""\n"
          else:
            remove_conditions = self.combine_conditions(item_conditions, old_group_conditions.get(var.name, None))
            if remove_conditions:
              new_group_conditions[var.name] = self.combine_conditions(new_group_conditions[var.name], "not " + remove_conditions)
            else:
              new_group.pop(var.name)
              new_group_conditions.pop(var.name)
        else:
          new_conditions = self.combine_conditions(
            item_conditions,
            old_group_conditions.get(var.name, None),
          )

          if var.name in new_group:
            new_conditions = self.combine_conditions(
              new_conditions,
              new_group_conditions[var.name],
              intersect=False  # We use OR (intersect) to merge groups
            )
          else:
            new_group[var.name] = var, level

          new_group_conditions[var.name] = new_conditions

    for var, level in new_group.values():
      # Declare the variables if new
      if var.name in GROUPS["all"]:
        new_group[var.name] = GROUPS["all"][var.name]
      else:
        replacement_text += f"{sym_type} {var.name}{var.sets} {var.label} //;\n"
        if DUMMY_SUFFIX:
          replacement_text += f"SET {var.name}{DUMMY_SUFFIX}{var.sets};\n"
        new_group[var.name] = var

      # Set levels if a value is given
      if level:
        conditions = self.merge_conditions(new_group_conditions[var.name])
        replacement_text += f"{var.name}{L}{var.sets}{conditions} = {level};\n"

    replacement_text += "$onlisting\n"
    
    GROUPS[group_name] = new_group
    GROUPS["all"] = Group(new_group, **GROUPS["all"])
    CONDITIONS[group_name] = new_group_conditions
    CONDITIONS["all"] = {k: None for k in GROUPS["all"]}

    return replacement_text

  def par_group_define(self, match, text):
    return self.group_define(match, text, sym_type="Parameter")

  def set_group_define(self, match, text):
    return self.group_define(match, text, sym_type="Set")

  def define_function(self, match, text):
    """
    Define gamY macros (multiline global variable)
    Example:
    $FUNCTION InflationCorrection:
      $LOOP G_prices
        {name}.l{sets} = {name}.l{sets} * inf_factor[t];
      $ENDLOOP
    $ENDFUNCTION
    # Remove inflation and growth correction
    %InflationCorrection%
    """
    id_, name, args, expression = match.groups()
    if not id_:
      id_ = " "
    if f"$FUNCTION{id_}" in expression:
      self.error(f"Nested FUNCTION definitions must be identified with id numbers (e.g. $FUNCTION1 .. $ENDFUNCTION1): {match.groups()[:-1]}")
    self.user_functions[name] = Function(name, args, expression)
    return f"\n# ----- gamY: Define function: {name} -----"

  def for_loop(self, match, text):
    """
    Parse $FOR command.
    Loops over the expression parsing the for loop with python syntax.
    Examples:
    $FOR &i in range(1,5):
      a("t&i") = &i**2;
    $ENDFOR
    $FOR &i in ["Grane", "joao", "Martin"]:
      $import &i.gms
    $ENDFOR
    For loops can be nested by adding an id number after the command
    E.g.
    $FOR {p1} in ["Lob"]:
      $FOR1 {p2} in ["Bob"]:
        $FOR2 {value} in ["Loblaw"]:
          set {p1}{p2} /{value}/;
        $ENDFOR2
      $ENDFOR1
    $ENDFOR
    """
    id_ = match.group(1)
    if not id_:
        id_ = " "
    iterators = self.parse(match.group(2)).replace(" ", "").split(",")
    iterable = self.parse(match.group(3))
    expression = match.group(4)
    replacement_text = ""
    if f"$FOR{id_}" in expression:
      self.error(f"Nested FOR loops must be identified with id numbers (e.g. $FOR1 .. $ENDFOR1): {match.groups()[:-1]}")
    try:
      for i in eval(iterable):
        replacement_text += expression
        if len(iterators) == 1:
          replacement_text = replacement_text.replace(iterators[0], str(i))
        else:
          for index, iterator in enumerate(iterators):
            replacement_text = replacement_text.replace(iterator, str(i[index]))
    except SyntaxError:
      self.error(f"Failed to evalute: {iterable}""\n It is not a proper iterable.")
    return replacement_text

  def loop(self, match, text):
    """
    Parse $Loop command.
    Loops over the expression once for each variable in the group, or each equation in a block
    replacing
    "{NAME}" with the name of the variable or equation
    "{SETS}" with the sets of the variable or equation
    "{CONDITIONS}" with any conditionals (subsetting) of the equation. E.g. "$ax0(a)"
    "{LHS}" with the left hand side of an equation.
    "{RHS}" with the right hand side of an equation.
    The {$} operator can be used to modify sets
    Examples:
    # Remove a and add t to all variables
    $LOOP G_endo
      {name}.l{sets}{$}[+t,-a] = {name}.l{sets};
    $ENDLOOP
    # Replace 'a' with 'a0'
    $LOOP G_endo
      {name}.l{sets}{$}[<a0>a] = {name}.l{sets};
    $ENDLOOP
    # Replace 'a' with 'a0(a)'
    $LOOP G_endo
      {name}.l{sets}{$}[a0[a]] = {name}.l{sets};
    $ENDLOOP
    """
    id_ = match.group(1)
    if not id_:
        id_ = " "
    iterable_name = self.parse(match.group(2))  #  Name of group or block to be iterated over
    expression = match.group(3)  # Inside of loop
    # expression = self.round_parentheses(expression)
    if f"$LOOP{id_}" in expression:
      self.error(f"Nested loops must be identified with id numbers (e.g. $LOOP1 .. $ENDLOOP1): {match.groups()[:-1]}")

    replacement_text = f"\n# ----- gamY: Loop over {iterable_name} -----\n"

    if iterable_name in self.groups:
      replacement_text += self.loop_over_variables(
        expression,
        self.groups[iterable_name].values(),
        group_conditions=self.groups_conditions[iterable_name]
      )
    elif iterable_name in self.par_groups:
      replacement_text += self.loop_over_variables(
        expression,
        self.par_groups[iterable_name].values(),
        group_conditions=self.par_groups_conditions[iterable_name]
      )
    elif iterable_name in self.set_groups:
      replacement_text += self.loop_over_variables(
        expression,
        self.set_groups[iterable_name].values(),
        group_conditions=self.set_groups_conditions[iterable_name]
      )
    elif iterable_name in self.blocks:
      replacement_text += self.loop_over_equations(expression, self.blocks[iterable_name].values())
    elif iterable_name in self.groups["all"]:
      #  If looping over a single variable, we put it in a list to be iterable
      replacement_text += self.loop_over_variables(expression, [self.groups["all"][iterable_name]], {iterable_name: None})
    elif iterable_name in self.equations:
      #  If looping over a single equation, we put it in a list to be iterable
      replacement_text += self.loop_over_equations(expression, [self.equations[iterable_name]])
    else:
      self.error('"{}" is not a block, group, or variable and cannot be looped over.'.format(iterable_name))

    return replacement_text

  def loop_over_variables(self, expression, variables, group_conditions):
    """
    Loop over the variables in a group (used by $LOOP command)
    """

    iter_patterns = {
      "name": re.compile(r"{NAME}", re.IGNORECASE),
      "sets": re.compile(r"{SETS?}?(\{\$\}\[[^$=\n;{}]+\])?", re.IGNORECASE),
      "conditions": re.compile(r"{(SUBSET|CONDITION)S?}", re.IGNORECASE),  # "SUBSETS" will be depreciated
      "text": re.compile(r"{text}", re.IGNORECASE),
    }

    replacement_text = ""
    for variable in variables:
      sub = expression

      # Replace sets
      while iter_patterns["sets"].search(sub):
        filter = iter_patterns["sets"].search(sub).group(1)
        if filter:
          sets = variable.sets[1:-1].replace(" ", "").split(",")  # Remove parentheses and whitespace and splits sets into list
          add_sets, subtract_sets, replace_sets = [], [], []
          for f in filter[4:-1].replace(" ", "").split(","):
            if f[0] == "+":
              add_sets.append(f[1:])
            elif f[0] == "-":
              subtract_sets.append(f[1:])
            else:
              replace_sets.append(f)

          filtered_sets = sets + add_sets
          for s in sets:
            for f in add_sets:
              if f"[{f}]" in s:  # If the variable's set is a subset of the add set, or if the add set is already present, don't add it
                pass
              elif "[{}]".format(s) in f:
                filtered_sets.remove(s)

            for f in subtract_sets:
              if s == f or "[{}]".format(f) in s:  # If the variable's sets is a subset of the filter set, the set should be removed
                filtered_sets.remove(s)

            for f in replace_sets:
              if "<{}>".format(s) in f:  # Replace set surrounded by < > with the filter
                filtered_sets[filtered_sets.index(s)] = f.replace("<{}>".format(s), "")
              elif "[{}]".format(s) in f:
                filtered_sets.remove(s)
                filtered_sets.append(f)

          sets = []
          for s in filtered_sets:
            if s != "" and s not in sets:
              sets.append(s)
          sets = "[" + ",".join(sets) + "]"
          if sets == "[]":
            sets = ""
          sub = iter_patterns["sets"].sub(sets, sub, count=1)
        else:
          sub = iter_patterns["sets"].sub(variable.sets, sub, count=1)

      # Replace name
      sub = iter_patterns["name"].sub(variable.name, sub)

      # Replace conditions
      if group_conditions[variable.name]:
        sub = iter_patterns["conditions"].sub(group_conditions[variable.name], sub)
      else:
        sub = iter_patterns["conditions"].sub("(1)", sub)

      # Replace text
      sub = iter_patterns["text"].sub(variable.label, sub)

      replacement_text += sub
    return replacement_text

  def loop_over_equations(self, expression, equations):
    """
    Loop over the equations in a block (used by $LOOP command)
    """

    iter_patterns = {
      "name": re.compile(r"{NAME}", re.IGNORECASE),
      "sets": re.compile(r"{SETS}?(\{\$\}\[[^=\n;{}]+\])?", re.IGNORECASE),
      "conditions": re.compile(r"{(SUBSET|CONDITION)S?}", re.IGNORECASE),
      "LHS": re.compile(r"{LHS}", re.IGNORECASE),
      "RHS": re.compile(r"{RHS}", re.IGNORECASE)
    }

    replacement_text = ""
    for eq in equations:
      # All equations must have a subset enclosed in parentheses to allow adding to the subset using and/or.
      # A subset of (1) is added if none exists
      if eq.conditions == "":
        eq.conditions = "(1)"
      if eq.conditions[0] == "$":
        eq.conditions = eq.conditions[1:]
      if eq.conditions[0] != "[":
        eq.conditions = "[" + eq.conditions + "]"
      eq.conditions = eq.conditions

      sub = expression

      # Replace sets
      while iter_patterns["sets"].search(sub):
        filter = iter_patterns["sets"].search(sub).group(1)
        if filter:
          sets = eq.sets[1:-1].replace(" ", "").split(",")  # Remove parentheses and whitespace and splits sets into list
          add_sets, subtract_sets, replace_sets = [], [], []
          for f in filter[4:-1].replace(" ", "").split(","):
            if f[0] == "+":
              add_sets.append(f[1:])
            elif f[0] == "-":
              subtract_sets.append(f[1:])
            else:
              replace_sets.append(f)

          filtered_sets = sets + add_sets
          for s in sets:
            for f in add_sets:
              if "[{}]".format(f) in s:  # If the variable's set is a subset of the add set, or if the add set is already present, don't add it
                filtered_sets.remove(f)
              elif "[{}]".format(s) in f:
                filtered_sets.remove(s)

            for f in subtract_sets:
              if s == f or "[{}]".format(f) in s:  # If the variable's sets is a subset of the filter set, the set should be removed
                filtered_sets.remove(s)

            for f in replace_sets:
              if "<{}>".format(s) in f:  # Replace set surrounded by < > with the filter
                filtered_sets[filtered_sets.index(s)] = f.replace("<{}>".format(s), "")
              elif "[{}]".format(s) in f:
                filtered_sets.remove(s)
                filtered_sets.append(f)

          sets = []
          for s in filtered_sets:
            if s != "" and s not in sets:
              sets.append(s)
          sets = "[" + ",".join(sets) + "]"
          if sets == "[]":
            sets = ""
          sub = iter_patterns["sets"].sub(sets, sub, count=1)
        else:
          sub = iter_patterns["sets"].sub(eq.sets, sub, count=1)

      for key, pattern in iter_patterns.items():
        sub = pattern.sub(eq.__dict__[key], sub)
      replacement_text += sub
    return replacement_text

  def eval_python(self, match, text):
    """
    Evaluate python code
    """
    code = dedent(match.group(1))
    return eval(code)

  def display(self, match, text, ignore_conditionals=False):
    """
    Display group in LST file
    """
    content = match.group(1)

    replacement_text = ""
    report = []

    # We use the group command to define a temporary group.
    # This ensures that the display syntax is identical to the GROUP command.
    self.group_define(MockMatch("", "temp_display_group", content), text)

    for var in self.groups["temp_display_group"].values():
      conditions = self.groups_conditions["temp_display_group"][var.name]
      if ignore_conditionals or not conditions:
        conditions = "(1)"

      priority = len(var.sets.split(","))
      p_name = "report__" + var.sets.replace(",", "_").replace(" ", "")[1:-1]
      if (priority, p_name) not in report:
        if var.sets == "":
          replacement_text += f"parameter {p_name};\n"
        else:
          replacement_text += f"parameter {p_name}[{var.sets[1:-1]}, *];\n"
        heappush(report, (priority, p_name))
      EPS_or_zero = "0";
      sets = var.sets[1:-1]
      if var.sets == "":
        replacement_text += "{p_name}('{var.name}')${conditions} = {var.name}.L + {EPS_or_zero};\n".format(**locals())
      else:
        replacement_text += "{p_name}({sets}, '{var.name}')${conditions} = {var.name}.L{var.sets} + {EPS_or_zero};\n".format(**locals())

    for priority, p_name in report:
      replacement_text += "display {};\n".format(p_name)
      replacement_text += "Option Clear={};\n".format(p_name)

    return replacement_text

  def display_all(self, match, text):
    return self.display(match, text, ignore_conditionals=True)

  def solve(self, match, text):
    """
    """
    content = match.group(1)
    model_name = "temp_model_{}".format(self.model_counter)
    self.model_counter += 1
    replacement_text = self.model_define(MockMatch(model_name, content), text)
    replacement_text += "Solve {} using CNS;".format(model_name)
    return replacement_text

  #  $FIX $UNFIX
  def fix_unfix(self, match, text, lower_bound="-inf", upper_bound="inf", level_value=None):
    """
    Return string with FIX and UNFIX commands replaced by GAMS code.
    Fix/Unfix commands
    Can be used either on a group, a variable, or ALL (collection of all variables defined with group commands)
    The command can be limited to a subset in parenthesis
    Examples:
      $FIX ALL;
      $UNFIX group1(subset1[t]);
      $FIX var1;
      $FIX(0) J;  #  Set all J terms to 0
      $UNFIX(0, inf) G_prices;  #  Unfix group and set lower bound to zero
    """
    command = match.group(1).lower()
    bounds = match.group(2)
    content = self.remove_comments(match.group(3))
    if command == "$fix" and bounds:
      level_value = bounds
    if command == "$unfix" and bounds:
      lower_bound, upper_bound = bounds.split(",")

    replacement_text = f"\n# ----- gamY: {self.comment_out(match.group(0))} -----\n$offlisting\n"

    # We use the group command to define a temporary group.
    # This ensures that the syntax for FIX/UNFIX is identical to the GROUP command.
    self.group_define(MockMatch("", "temp_fix_unfix_group", content), text)

    for var in self.groups["temp_fix_unfix_group"].values():
      conditions = self.merge_conditions(self.groups_conditions["temp_fix_unfix_group"][var.name])

      if (command == "$fix"):
        if level_value:
          replacement_text += "{var.name}.FX{var.sets}{conditions} = {level_value};\n".format(**locals())
        else:
          replacement_text += "{var.name}.FX{var.sets}{conditions} = {var.name}.L{var.sets};\n".format(**locals())
      elif command == "$unfix":
        replacement_text += "{var.name}.lo{var.sets}{conditions} = {lower_bound};\n".format(**locals())
        replacement_text += "{var.name}.up{var.sets}{conditions} = {upper_bound};\n".format(**locals())

    replacement_text += "$onlisting\n"

    return replacement_text


  @staticmethod
  def dedent_dollar(text):
    pattern = re.compile(r"^\s+\$", re.MULTILINE)
    return pattern.sub("$", text)

  @staticmethod
  def combine_conditions(*args, intersect=True):
    """Combine multiple conditions such that '$t0[t]' and '$a0[t]' becomes '$((t0[t]) and (a0[t]))' """

    #  Create list of non-empty conditions
    conditions = []
    for subset in args:
      if subset:
        subset = subset.replace("$", "", 1)
        if subset not in conditions:
          conditions.append(subset)
      elif not intersect:
        return ""  # Combining an empty condition using OR removes all conditions

    if not conditions:
      return ""
    elif intersect:
      combined = " and ".join(conditions)
    else:
      combined = " or ".join(conditions)
    if is_enclosed(combined):
      return combined
    else:
      return "({})".format(combined)

def is_enclosed(expression):
  """
  Check if the the expression is enclosed in brackets
  >>> is_enclosed('(foo)(bar)')
  False
  >>> is_enclosed('((foo)(bar))')
  True
  >>> is_enclosed('[[foo]{bar}]')
  True
  """
  brackets = ["()", "[]", "{}"]
  if expression[0] + expression[-1] not in brackets:
    return False
  for br in brackets:
    open_count = 0
    for char in expression[1:-1]:
      if char == br[0]:
        open_count += 1
      if char == br[1]:
        open_count -= 1
      if open_count < 0:
        return False
  return True


def find_gams():
    """
    Find GAMS executable
    """
    gams_path = shutil.which("GAMS") or shutil.which("gams") or os.environ.get("GAMS") or os.environ.get("gams")
    
    if not os.path.isfile(gams_path):
      sys.exit("ERROR: gamY could not find GAMS. Set path to gams executable as environmental variable with variable name GAMS (or gams)")
    
    return gams_path


def run(file_path, r=None, s=None, shell=False, **kwargs):
  """
  Run a GAMS file after precompiling it with optional adjustments.

  Parameters:
  - file_path (str): Path to the GAMS file to be precompiled and executed.
  - r (str, optional): Optional argument to read the gamY data structure.
  - s (str, optional): Optional argument to save the gamY data structure.
  - shell (bool, optional): If True, the command will be executed through the shell.
  - **kwargs: Additional keyword arguments to be passed to the GAMS executable.

  Raises:
  - AssertionError: If the GAMS executable is not found or if the provided path is not an executable.
  - Exception: If there is an error in GAMS execution.
  """

  start_time = timer()

  precompiler = Precompiler(file_path)

  expanded_dir = os.path.join(precompiler.file_dir, "Expanded")
  parsed_file_path = os.path.join(expanded_dir, precompiler.file_name.replace(".gms", ".gmy"))

  call_parameters = [find_gams(), parsed_file_path]

  if r is not None: # Read gamY data structure if gamY is called with r= argument and pass r= argument to GAMS
    precompiler.read(r)
    call_parameters.append(f"r={r}")

  for k, v in kwargs.items():
    precompiler.locals[k] = v
    call_parameters.append(f"--{k}={v}")

  #  Parse file using recursive descent
  text = precompiler()

  if s is not None: # Save gamY data structure if gamY is called with s= argument and pass s= argument to GAMS
    precompiler.save(s)
    call_parameters.append(f"s={s}")

  #  Save pre-compiled GAMS file in 'Expanded' folder
  os.makedirs(expanded_dir, exist_ok=True)
  with open(parsed_file_path, 'w') as f:
    f.write(text)

  compilation_time = timer() - start_time

  #  Run file using GAMS (path needs to be set for system or user)
  prev_line = ""
  call_parameters += [
    "LO=3",
    f"Workdir={precompiler.file_dir}",
    f"CurDir={precompiler.file_dir}",
    "ErrMsg=1",
    f"O={precompiler.list_file_path}",
    "pageSize=0",
    "pageWidth=9999",
  ]
  process = subprocess.Popen(
    call_parameters,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    universal_newlines=True, shell=shell
  )

  while process.poll() is None:
    for line in iter(process.stdout.readline, ""):
      if not (line[:3] == "---" and prev_line[:10] == line[:10]):  # For almost identical lines, only print the last one
        sys.stdout.write(prev_line)
        sys.stdout.flush()
      prev_line = line
  sys.stdout.write(prev_line)
  sys.stdout.flush()

  print(get_status_from_lst(precompiler.list_file_path))

  execution_time = timer() - start_time - compilation_time
  print(f"Precompiler time: {compilation_time:.2f} seconds")
  print(f"Execution time: {execution_time:.2f} seconds")
  print(f"Total run time: {execution_time + compilation_time:.2f} seconds")

  process.terminate() # terminate process
  process.wait()
  
  if process.returncode != 0:
    raise Exception(f"Error in GAMS execution. Return code: {process.returncode}")

  if shell:
    sys.exit(process.returncode)
  
  return precompiler


status_message_patterns = [
  "****",
  "Model Statistics    SOLVE",
  "SINGLE =E|X= EQUS",
]

error_message_patterns = [
  "USER ERROR(S) ENCOUNTERED",
  "** Infeasible solution to a square system",
  "Locally Infeasible",
  "Terminated By Solver",
  "Solver Failure",
  "Intermediate Infeasible",
]

status_message_exclude_patterns = [
  "**** REPORT FILE SUMMARY",
  "**** FILE SUMMARY",
  "**** WARNING - COMPILER OPTIONS ARE NON DEFAULT",
]

def get_status_from_lst(list_file_path):
  """Extract status messages from the listing file (lines starting with ****)"""
  with open(list_file_path, 'r') as f:
      status_messages = [
          line for line in f
          if any(line.startswith(pattern) for pattern in status_message_patterns + error_message_patterns)
          and not any(line.startswith(exclude_pattern) for exclude_pattern in status_message_exclude_patterns)
      ]
  return "\n" + "".join(status_messages)

def print_status(gams_file):
  print(get_status_from_lst(get_lst_path(gams_file)))


def cmd_call():
  """Wrapper for calling gamY from command line."""
  # Translate command line arguments to key-word arguments for the run function
  cmd_args = sys.argv
  kwargs = {"file_path": cmd_args[1]}

  # Read optional command line arguments
  for arg in cmd_args[2:]:
    # Remove the '--' prefix if it exists
    if arg.startswith("--"):
      arg = arg[2:]
    # Split the argument into key and value
    key, value = arg.split("=", 1)
    kwargs[key] = value

  return run(**kwargs, shell=True)


if __name__ == "__main__":
  cmd_call()