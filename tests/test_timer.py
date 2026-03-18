"""Tests for src.utils.timer — timing utility."""

import time

from src.utils.timer import Timer


class TestTimer:
    def test_basic_timing(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04
        assert t.elapsed < 1.0

    def test_named_timer(self, capsys):
        with Timer("test_op"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert "test_op" in captured.out

    def test_elapsed_during(self):
        t = Timer()
        t.__enter__()
        time.sleep(0.02)
        assert t.elapsed >= 0.01
        t.__exit__(None, None, None)

    def test_elapsed_grows(self):
        with Timer() as t:
            e1 = t.elapsed
            time.sleep(0.02)
            e2 = t.elapsed
        assert e2 > e1
