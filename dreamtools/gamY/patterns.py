import re

# Tidbits used to make regex-patterns more readable
open_bracket = r"[\(\[]"
close_bracket = r"[\)\]]"
brackets = r"[\(\)\[\]]"
no_brackets = r"[^\(\)\[\]]"
ident = r"[A-Za-z0-9_]{1,62}" # A valid symbol in GAMS is a letter followed by up to 62 letters, numbers or underscores. gamY also accepts starting with a number or underscore.
linebreak = r"(?:\r\n|\r|\n)"

# Top level patterns for all commands. Used to recursively parse script.
PATTERN_STRINGS = {
    # "env_variable": r"""(?:[^\s\*][^\n]*)?(\%(\S+?)\%)""",
    "env_variable": fr"""\%({ident})\%""",

    "user_function": fr"""
        @({ident})  # Function name
        \(
            ([^)]*?)  # Arguments
        \)
    """,

    "set": fr"""

                \$set(global|local|)
                \s*
                ({ident})
                \s+
                ([^\s\;]+)
            """,

    "eval": fr"""

                \$eval(global|local|)
                \s+
                ({ident})
                \s+
                ([^\;]+)
            """,

    "import": r"""
        \$Import
        \s+
        ([^\s;]+)                # File name
    """,

    "block": fr"\$Block\s+({ident})\s*(\$.+?)?\s*{linebreak}(.*?)\$EndBlock",

    "group": fr"\$Group(\+?)\s+({ident})\s+(.*?;)",  # 1) +? 2) name, 3) content

    "pgroup": fr"\$PGroup(\+?)\s+({ident})\s+(.*?;)",

    "display": r"""
        \$Display\s
            (.+?;)      # Variables and groups to be displayed
    """,

    "display_all": r"""
        \$Display_all
            (.+?;)      # Variables and groups to be displayed
    """,

    "model": fr"\$Model\s+({ident})\s+(.*?);",

    "solve": r"\$Solve\s+(.*?);",

    "fix": fr"""
                        (\$(?:UN)?FIX)          # Fix or unfix command
                        (?:{open_bracket} (
                            -? (?:\d+\.?\d*|INF|EPS) (?: \,\s*-?(?:\d+\.?\d*|INF|EPS) )?       # Optional arguments
                        ) {close_bracket})?
                        \s+
                        (.+?;)                  # Content
                    """,


    "if": r"""
                    \$If(?P<if_id>\d*)\s+
                    ([^:]*?)             # Condition
                    \:
                    (.*?)
                    \$EndIF(?P=if_id)\b
                """,

    "define_function": fr"""
                        \$Function(?P<function_id>\d*)\s+
                        ({ident})        # Name $1
                        {open_bracket}([^)\]]*){close_bracket}  # Arguments  $2
                        (?:\:)?\s+
                        (.*?)         # Expression $3
                        \$EndFunction(?P=function_id)\b
    """,

    "for_loop": r"""
                        \$For(?P<for_id>\d*)
                        \s+
                        (.+?) # The iterator
                        \s+
                        in
                        \s+
                        ([^\:]*?)  # The iterable
                        :
                        (.*?)
                        \$EndFor(?P=for_id)\b
    """,

    "loop": fr"""
                        \$Loop(?P<loop_id>\d*)
                        \s*
                        ({ident})               # Group name
                        [:]
                        (.*?)
                        \$EndLoop(?P=loop_id)\b
                    """,

    "replace": fr"""
                        \$Replace
                        {open_bracket}
                        ('.*?'|".*?")       # String to find
                        ,\s*
                        ('.*?'|".*?")       # Replacement string
                        (,\s*\d+)?          # Max replacements
                        {close_bracket}
                        (.*?)
                        \$EndReplace
                    """,

    "regex": fr"""
                        \$Regex(?P<regex_id>\d*)
                        {open_bracket}\s*
                        ('.*?'|".*?"|[^,]+)       # String to find
                        ,\s*
                        ('.*?'|".*?")             # Replacement string
                        (,\s*\d+)?                # Max replacements
                        {close_bracket}
                        (.*?)
                        \$EndRegex(?P=regex_id)\b
                    """,

    "eval_python": fr"""
        \$EvalPython
        (.*?) # Code
        \$EndEvalPython
    """
}

# Compile regex patterns
PATTERNS = {k: re.compile(v, re.VERBOSE | re.IGNORECASE | re.MULTILINE | re.DOTALL) for k, v in PATTERN_STRINGS.items()}
# Create combined pattern that matches any of the patterns
PATTERNS["Any"] = re.compile("|".join(PATTERN_STRINGS.values()), re.VERBOSE | re.IGNORECASE | re.MULTILINE | re.DOTALL)


PATTERNS["TopDown"] = re.compile("|".join((  # Remember to also add these to the list in gamY
        PATTERN_STRINGS["if"],
        PATTERN_STRINGS["for_loop"],
        PATTERN_STRINGS["loop"],
        PATTERN_STRINGS["define_function"],
    )),
    re.VERBOSE | re.IGNORECASE | re.MULTILINE | re.DOTALL
)
