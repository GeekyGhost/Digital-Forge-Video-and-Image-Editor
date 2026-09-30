"""The stage split, which is the difference between a measurement and an anecdote.

A 7:11 retake and a 7:17 WanGP step count are not comparable numbers if one includes model
loading and the other does not -- and comparing them anyway is how you conclude the opposite of
the truth. This exists so our own runs never have that ambiguity.
"""

from __future__ import annotations

import pytest

from aiv_runtime.run_timing import StageTimer


class _Clock:
    """A clock that only moves when the test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _timer(clock: _Clock) -> StageTimer:
    return StageTimer("ltx-retake", clock=clock)


def test_stages_record_the_gap_since_the_previous_boundary():
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(12.0)
    assert timer.stage("import") == pytest.approx(12.0)
    clock.advance(240.0)
    assert timer.stage("load") == pytest.approx(240.0)
    clock.advance(168.0)
    assert timer.stage("denoise+decode") == pytest.approx(168.0)
    assert [name for name, _ in timer.stages] == ["import", "load", "denoise+decode"]


def test_the_summary_names_each_share():
    """The share is the point: 'load 4:00 (57%)' is what decides whether to build a resident worker."""
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(60.0)
    timer.stage("load")
    clock.advance(60.0)
    timer.stage("denoise+decode")
    line = timer.summary()
    assert "load 1:00.0 (50%)" in line
    assert "denoise+decode 1:00.0 (50%)" in line
    assert "measured 2:00.0" in line


def test_seconds_under_a_minute_stay_seconds():
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(8.25)
    timer.stage("encode")
    assert "encode 8.2s" in timer.summary()


def test_share_answers_the_resident_worker_question():
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(331.0)
    timer.stage("load")
    clock.advance(100.0)
    timer.stage("denoise+decode")
    # 331 of 431 seconds is load: a resident worker would remove three quarters of this run.
    assert timer.share("load") == pytest.approx(331 / 431)
    assert timer.share("denoise+decode") == pytest.approx(100 / 431)


def test_an_unrecorded_stage_has_no_share_rather_than_raising():
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(5.0)
    timer.stage("load")
    assert timer.share("denoise+decode") == 0.0


def test_a_run_that_recorded_nothing_says_so():
    timer = _timer(_Clock())
    assert "nothing recorded" in timer.summary()
    assert timer.share("load") == 0.0


def test_total_is_wall_clock_not_the_sum_of_stages():
    """Time between the last boundary and now belongs to the run even though no stage claimed it."""
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(10.0)
    timer.stage("load")
    clock.advance(7.0)
    assert timer.total == pytest.approx(17.0)
    assert sum(duration for _, duration in timer.stages) == pytest.approx(10.0)


def test_report_writes_where_the_other_diagnostics_go(capsys):
    clock = _Clock()
    timer = _timer(clock)
    clock.advance(3.0)
    timer.stage("import")
    line = timer.report()
    captured = capsys.readouterr()
    assert captured.err.strip() == line
    assert captured.out == "", "stdout carries the JSON payload and nothing else"
    assert line.startswith("[ltx-retake] timing:")
