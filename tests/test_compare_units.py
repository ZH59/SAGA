#!/usr/bin/env python3
"""Unit tests for the response-time parsing and comparison logic in compare.py.

Each case has a known solution stated inline, so a reader can check the expected
value by hand. Run with:  python3 -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import compare  # noqa: E402


class TestReadRta(unittest.TestCase):
    """read_rta() must skip the header and key rows by (task, job)."""

    def _write(self, text):
        fh = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        fh.write(text)
        fh.close()
        return fh.name

    def test_parses_rows_and_skips_header(self):
        p = self._write(
            "Task ID, Job ID, BCCT, WCCT, BCRT, WCRT\n"
            "0, 0, 6, 11, 6, 11\n"
            "0, 1, 7, 12, 7, 12\n"
        )
        got = compare.read_rta(p)
        # Known solution: two jobs, header ignored, WCRT is the last column.
        self.assertEqual(len(got), 2)
        self.assertEqual(got[(0, 0)], (6, 11, 6, 11))
        self.assertEqual(got[(0, 1)][3], 12)

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(compare.read_rta("/nonexistent/nope.csv"), {})

    def test_short_rows_ignored(self):
        p = self._write("Task ID, Job ID\n0, 0\n1, 1, 2, 3, 4, 5\n")
        got = compare.read_rta(p)
        self.assertEqual(list(got), [(1, 1)])


class TestCompareRta(unittest.TestCase):
    """compare_rta() buckets GPU WCRT against the exact reference.

    exact   : equal
    loose   : GPU larger  -> sound but pessimistic
    unsound : GPU smaller -> a real defect
    """

    def test_all_exact(self):
        ref = {(0, 0): (1, 10, 1, 10), (0, 1): (1, 20, 1, 20)}
        got = compare.compare_rta(dict(ref), ref)
        self.assertEqual((got["exact"], got["loose"], got["unsound"]), (2, 0, 0))
        self.assertEqual(got["maxdev"], 0)

    def test_larger_bound_is_loose_not_unsound(self):
        ref = {(0, 0): (1, 10, 1, 10)}
        gpu = {(0, 0): (1, 10, 1, 13)}   # WCRT 13 > 10
        got = compare.compare_rta(gpu, ref)
        # Known solution: sound over-approximation, deviation 3.
        self.assertEqual((got["exact"], got["loose"], got["unsound"]), (0, 1, 0))
        self.assertEqual(got["maxdev"], 3)

    def test_smaller_bound_is_unsound(self):
        ref = {(0, 0): (1, 10, 1, 10)}
        gpu = {(0, 0): (1, 10, 1, 9)}    # WCRT 9 < 10  -> below the exact bound
        got = compare.compare_rta(gpu, ref)
        self.assertEqual((got["exact"], got["loose"], got["unsound"]), (0, 0, 1))

    def test_only_shared_keys_are_compared(self):
        ref = {(0, 0): (1, 10, 1, 10), (0, 1): (1, 10, 1, 10)}
        gpu = {(0, 0): (1, 10, 1, 10), (9, 9): (1, 10, 1, 10)}
        got = compare.compare_rta(gpu, ref)
        self.assertEqual(got["jobs"], 1)
        self.assertEqual(got["only_gpu"], 1)
        self.assertEqual(got["only_ref"], 1)


class TestPathResolution(unittest.TestCase):
    """Regression: every path argument must be made absolute.

    Both tools are launched with cwd set to the per-task-set scratch directory.
    If --work is left relative (its default is the relative "work"), the jobs
    path handed to the tools resolves against that scratch directory instead of
    the caller's cwd, and both fail with "cannot open jobs file".
    """

    def test_relative_work_dir_is_resolved(self):
        rel = Path("work")
        self.assertFalse(rel.is_absolute(), "precondition: default --work is relative")
        self.assertTrue(rel.resolve().is_absolute(),
                        "resolve() must yield an absolute path")

    def test_relative_path_breaks_when_cwd_is_the_scratch_dir(self):
        """Reproduce the exact failure mode the fix guards against.

        Layout mirrors real use: <base>/work/taskset_000/jobs.csv, with the path
        expressed relative to <base> as "work/taskset_000/jobs.csv". Both tools
        are launched with cwd=<base>/work/taskset_000, where that same relative
        string does NOT resolve -- which is what produced
        "Error: cannot open jobs file 'work/full/taskset_000/jobs.csv'".
        """
        import os
        with tempfile.TemporaryDirectory() as base:
            ts = Path(base) / "work" / "taskset_000"
            ts.mkdir(parents=True)
            (ts / "jobs.csv").write_text("Task ID, Job ID, BCCT, WCCT, BCRT, WCRT\n")

            relative = Path("work/taskset_000/jobs.csv")   # relative to base
            absolute = (Path(base) / relative).resolve()

            cwd = os.getcwd()
            try:
                os.chdir(base)
                self.assertTrue(relative.exists(), "resolves from the base directory")
                os.chdir(ts)                                # what the tools see
                self.assertFalse(relative.exists(),
                                 "the bug: relative path does not resolve from the scratch dir")
                self.assertTrue(absolute.exists(),
                                "the fix: an absolute path resolves from anywhere")
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
