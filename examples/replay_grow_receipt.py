#!/usr/bin/env python3
"""Replay the committed grow-tests receipt offline (no API key) — the real
Codestral middle is on disk; the pytest run is real and deterministic."""
from pathlib import Path
from vibe_fim.testgen import grow_tests, pytest_runner

MID = (Path(__file__).parent / "fixtures" / "grow_tests_codestral_middle.txt").read_text()
SUBJECT = ("def clamp(v, lo, hi):\n    return max(lo, min(v, hi))\n\n\n"
           "def test_clamp_within():\n    assert clamp(5, 0, 10) == 5\n")
r = grow_tests(SUBJECT, filename="test_clamp.py", fim_fn=lambda p, s: MID, runner=pytest_runner)
print(r.report())
assert r.additive, "receipt should be additive"
print("\nREPLAY OK: provably additive, old test stayed", r.outcomes_after["test_clamp_within"])
