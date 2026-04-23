"""
Zero-dependency test runner.

Equivalent to running `python3 -m pytest tests/`, but uses only the
standard-library `unittest` loader + runner so the test suite works in
reviewer environments without pytest installed.

Run from the project root:
    python3 tests/run_tests.py
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main() -> int:
    loader = unittest.TestLoader()
    # Discover any test_*.py in this directory, treated as function-style
    # pytest tests. We wrap plain `def test_*` functions into TestCases.
    suite = unittest.TestSuite()

    import importlib.util

    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        path = os.path.join(HERE, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        for attr in sorted(vars(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue

            # Wrap free function into a TestCase instance.
            # We construct a one-shot TestCase class so assertion failures
            # surface naturally through unittest's runner.
            def make_test(func):
                class _T(unittest.TestCase):
                    def runTest(self):
                        # support a single `tmp_path` kwarg like pytest fixture
                        import inspect
                        sig = inspect.signature(func)
                        if "tmp_path" in sig.parameters:
                            import tempfile
                            import pathlib
                            with tempfile.TemporaryDirectory() as td:
                                func(pathlib.Path(td))
                        else:
                            func()
                _T.__name__ = func.__name__
                return _T

            suite.addTest(make_test(fn)())

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
