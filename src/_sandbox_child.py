"""Runs one candidate `transform` function in a separate, short-lived process.

Protocol: reads {"code": str, "inputs": [[col, ...], ...]} as JSON on stdin and
writes {"load_error": str | null, "results": [{"value", "error"}, ...]} to stdout.

This file is executed as a script by sandbox.py and should not be imported.
The candidate runs with a whitelist of builtins, and `import` only returns
filtered views of a few pure-computation standard library modules. The parent
process adds a static check, a hard timeout, and (on POSIX) resource limits.
"""

import builtins
import io
import json
import sys
import types

# Pure computation modules only. Imported here so that the candidate's own
# imports are cheap and never trigger a fresh module load.
import calendar
import collections
import datetime
import decimal
import fractions
import functools
import itertools
import math
import re
import string
import textwrap
import time
import unicodedata

# datetime.strptime and time.strptime import this lazily through __import__, so
# the hook has to serve it. Candidate code can't name it: the static check in
# sandbox.py rejects imports of underscore-prefixed modules.
import _strptime

INTERNAL_MODULES = {"_strptime": _strptime}

ALLOWED_MODULES = {
    m.__name__: m
    for m in (calendar, collections, datetime, decimal, fractions, functools, itertools,
              math, re, string, textwrap, time, unicodedata)
}

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "callable", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
    "isinstance", "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError",
    "KeyError", "LookupError", "OverflowError", "RuntimeError", "StopIteration",
    "TypeError", "ValueError", "ZeroDivisionError",
]

MAX_VALUE_CHARS = 10_000

_module_views: dict[str, types.SimpleNamespace] = {}


def _module_view(name):
    """Public, non-module attributes of an allowed module.

    Handing out the real module would let `re.enum.sys.modules["os"]` and
    similar chains reach everything else in the interpreter.
    """
    if name not in _module_views:
        mod = ALLOWED_MODULES[name]
        attrs = {
            k: v for k, v in vars(mod).items()
            if not k.startswith("_") and not isinstance(v, types.ModuleType)
        }
        _module_views[name] = types.SimpleNamespace(**attrs)
    return _module_views[name]


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name in INTERNAL_MODULES:
        return INTERNAL_MODULES[name]
    if level != 0 or name not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed")
    return _module_view(name)


def _no_print(*args, **kwargs):
    pass


def _safe_builtins():
    safe = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    safe["print"] = _no_print
    return safe


def _describe(exc):
    return f"{type(exc).__name__}: {exc}"[:500]


def main():
    real_stdout = sys.stdout
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    # Anything the candidate manages to write goes nowhere.
    sys.stdout = io.StringIO()

    reply = {"load_error": None, "results": []}
    namespace = {"__builtins__": _safe_builtins(), "__name__": "candidate"}
    try:
        exec(compile(request["code"], "<candidate>", "exec"), namespace)
        fn = namespace.get("transform")
        if not callable(fn):
            raise NameError("no function named 'transform' was defined")
    except BaseException as exc:  # noqa: BLE001 - any failure to load is just a failed candidate
        reply["load_error"] = _describe(exc)
        fn = None

    if fn is not None:
        for cols in request["inputs"]:
            try:
                value = fn(*cols)
                if value is not None and not isinstance(value, str):
                    reply["results"].append(
                        {"value": None, "error": f"returned {type(value).__name__}, expected str"})
                    continue
                if value is not None and len(value) > MAX_VALUE_CHARS:
                    value = value[:MAX_VALUE_CHARS] + "...<truncated>"
                reply["results"].append({"value": value, "error": None})
            except BaseException as exc:  # noqa: BLE001
                reply["results"].append({"value": None, "error": _describe(exc)})

    real_stdout.write(json.dumps(reply, ensure_ascii=True))
    real_stdout.flush()


if __name__ == "__main__":
    main()
