"""Tests for provably-additive test generation. FIM + runner are injected; one
integration test uses the real pytest subprocess."""
from __future__ import annotations

import re

import pytest

from vibe_fim.testgen import grow_tests, pytest_runner

ORIGINAL = '''\
def add(a, b):
    return a + b


def test_add_basic():
    assert add(2, 3) == 5
'''


def _fake_fim(generated_body):
    return lambda prefix, suffix: generated_body


def _make_runner(flip=None, drop=None):
    """Fake runner: node-ids are the `def test_*` names in the text. `flip`/`drop`
    apply when a trigger substring is present, to simulate behavioral regressions."""
    flip, drop = flip or {}, drop or {}

    def run(text):
        out = {name: "passed" for name in re.findall(r"def (test_\w+)\(", text)}
        for trigger, changes in flip.items():
            if trigger in text:
                out.update({k: v for k, v in changes.items() if k in out})
        for trigger, ids in drop.items():
            if trigger in text:
                for i in ids:
                    out.pop(i, None)
        return out

    return run


def test_additive_happy_path():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("add_negatives():\n    assert add(-1, -1) == -2"),
                   runner=_make_runner())
    assert r.additive is True
    assert r.prefix_identical and r.parses is True
    assert r.ids_monotonic and r.green_preserved
    assert "test_add_negatives" in r.added_ids


def test_prefix_is_byte_identical():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("more():\n    assert add(0, 0) == 0"),
                   runner=_make_runner())
    assert r.text[:len(ORIGINAL)] == ORIGINAL        # P1, by construction
    assert "def test_add_basic():" in r.text          # the old test survives verbatim


def test_non_parsing_generation_is_not_additive():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("broken(:::"),
                   runner=_make_runner())
    assert r.parses is False
    assert r.additive is False


def test_behavioral_regression_blocks_additive():
    # a new test whose presence breaks an old one (bytes additive, behavior not)
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("evil():\n    assert True  # POISON"),
                   runner=_make_runner(flip={"POISON": {"test_add_basic": "failed"}}))
    assert r.green_preserved is False
    assert r.additive is False


def test_vanished_old_test_blocks_additive():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("x():\n    assert True  # SHADOW"),
                   runner=_make_runner(drop={"SHADOW": ["test_add_basic"]}))
    assert r.ids_monotonic is False                  # before not subset of after
    assert "test_add_basic" in r.removed_ids
    assert r.additive is False


def test_without_runner_is_not_proven_additive():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("y():\n    assert True"))
    assert r.ran is False
    assert r.additive is False                       # inserted, not proven
    assert "P3/P4 not evaluated" in r.report()


def test_report_flags_candidate_review():
    r = grow_tests(ORIGINAL, filename="t.py",
                   fim_fn=_fake_fim("z():\n    assert add(1, 1) == 2"),
                   runner=_make_runner())
    assert "CANDIDATES" in r.report() and "ADDITIVE" in r.report()


@pytest.mark.skipif(__import__("shutil").which("python3") is None, reason="needs python")
def test_integration_real_pytest_subprocess():
    # End to end with the REAL runner: a fake FIM appends a genuinely-passing test;
    # the real pytest subprocess proves the old test stayed green and the new one ran.
    r = grow_tests(
        ORIGINAL, filename="t.py",
        fim_fn=_fake_fim("add_commutes():\n    assert add(2, 3) == add(3, 2)"),
        runner=pytest_runner)
    assert r.ran is True
    assert r.additive is True, r.report()
    assert "test_add_basic" in r.before_ids
    assert "test_add_commutes" in r.added_ids
    assert r.outcomes_after["test_add_basic"] == "passed"
