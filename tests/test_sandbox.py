"""Checks that the sandbox runs ordinary code and refuses the obvious ways out.

Run with: python -m unittest discover -s tests
"""

import unittest

from src.sandbox import run_candidate


def run(code, *inputs):
    return run_candidate(code, [tuple(i) for i in inputs])


class OrdinaryCode(unittest.TestCase):
    def test_simple_transform(self):
        res = run("def transform(x):\n    return x.upper()", ["abc"], ["dé"])
        self.assertEqual(res.status, "ok")
        self.assertEqual([c.value for c in res.cases], ["ABC", "DÉ"])

    def test_allowed_imports(self):
        code = (
            "import re\n"
            "from datetime import datetime\n"
            "def transform(x):\n"
            "    d = datetime.strptime(x, '%Y-%m-%d')\n"
            "    return re.sub(r'\\s+', ' ', d.strftime('%d  %b %Y'))"
        )
        res = run(code, ["2024-03-05"])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.cases[0].value, "05 Mar 2024")

    def test_multiple_columns_and_none(self):
        code = "def transform(a, b):\n    return None if a is None else a + '-' + b"
        res = run(code, ["x", "y"], [None, "y"])
        self.assertEqual([c.value for c in res.cases], ["x-y", None])

    def test_per_case_exception_is_a_failed_case(self):
        res = run("def transform(x):\n    return x.split(',')[1]", ["a,b"], ["nocomma"])
        self.assertEqual(res.status, "ok")
        self.assertTrue(res.cases[0].ok)
        self.assertIn("IndexError", res.cases[1].error)

    def test_non_string_return_is_an_error(self):
        res = run("def transform(x):\n    return len(x)", ["abc"])
        self.assertIn("returned int", res.cases[0].error)

    def test_print_does_not_corrupt_output(self):
        res = run("def transform(x):\n    print('noise')\n    return x", ["a"])
        self.assertEqual(res.cases[0].value, "a")

    def test_timeout(self):
        res = run("def transform(x):\n    while True:\n        pass", ["a"])
        self.assertEqual(res.status, "timeout")

    def test_syntax_error_rejected(self):
        self.assertEqual(run("def transform(x) return x", ["a"]).status, "rejected")

    def test_missing_transform_rejected(self):
        self.assertEqual(run("def f(x):\n    return x", ["a"]).status, "rejected")


class EscapeAttempts(unittest.TestCase):
    def assertBlocked(self, code):
        res = run(code, ["a"])
        blocked = res.status in ("rejected", "load_error") or not res.cases[0].ok
        self.assertTrue(blocked, f"not blocked: {res}")

    def test_import_os(self):
        self.assertBlocked("import os\ndef transform(x):\n    return os.getcwd()")

    def test_import_inside_function(self):
        self.assertBlocked("def transform(x):\n    import subprocess\n    return 'x'")

    def test_open(self):
        self.assertBlocked("def transform(x):\n    return open('f.txt').read()")

    def test_eval(self):
        self.assertBlocked("def transform(x):\n    return eval('1+1')")

    def test_dunder_walk(self):
        self.assertBlocked("def transform(x):\n    return str(().__class__.__base__.__subclasses__())")

    def test_builtins_name(self):
        self.assertBlocked("def transform(x):\n    return str(__builtins__)")

    def test_module_attribute_chain(self):
        # re imports enum, and enum imports sys; the real module would expose both.
        self.assertBlocked("import re\ndef transform(x):\n    return str(re.enum.sys.modules)")

    def test_calendar_sys(self):
        self.assertBlocked("import calendar\ndef transform(x):\n    return str(calendar.sys)")

    def test_generator_frame(self):
        self.assertBlocked(
            "def transform(x):\n"
            "    g = (i for i in [1])\n"
            "    return str(g.gi_frame.f_back.f_globals)"
        )

    def test_private_module_import(self):
        self.assertBlocked("import _strptime\ndef transform(x):\n    return str(_strptime.re)")

    def test_getattr_string(self):
        self.assertBlocked("def transform(x):\n    return getattr(x, '__class__').__name__")


if __name__ == "__main__":
    unittest.main()
