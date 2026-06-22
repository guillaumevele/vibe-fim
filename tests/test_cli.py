"""CLI tests. The Codestral backend is monkeypatched — no network."""
from __future__ import annotations

import json

import pytest

from vibe_fim import cli


def _fake_backend(middle):
    # The CLI wraps the backend in functools.partial(..., model=, max_tokens=,
    # temperature=), so accept and ignore those kwargs.
    return lambda prefix, suffix, **kwargs: middle


PY = "def f():\n    # X\n    pass\n    # Y\n    return 1\n"


def test_dry_run_writes_diff_not_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    y = 2\n    "))
    f = tmp_path / "m.py"
    f.write_text(PY)
    code = cli.main(["patch", "--file", str(f), "--before", "# X", "--after", "# Y", "--dry-run"])
    assert code == 0
    assert f.read_text() == PY            # untouched
    assert "+    y = 2" in capsys.readouterr().out


def test_write_path_mutates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    y = 2\n    "))
    f = tmp_path / "m.py"
    f.write_text(PY)
    code = cli.main(["patch", "--file", str(f), "--before", "# X", "--after", "# Y"])
    assert code == 0
    assert "y = 2" in f.read_text()
    assert f.read_text().startswith("def f():\n    # X")


def test_missing_anchor_exit_1(tmp_path, monkeypatch):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("x"))
    f = tmp_path / "m.py"
    f.write_text(PY)
    assert cli.main(["patch", "--file", str(f), "--before", "# NOPE", "--after", "# Y"]) == 1


def test_unreadable_file_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("x"))
    missing = tmp_path / "nope.py"
    assert cli.main(["patch", "--file", str(missing), "--before", "# X", "--after", "# Y"]) == 2


def test_parse_failure_refuses_write_exit_3(tmp_path, monkeypatch):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    return (((\n    "))
    f = tmp_path / "m.py"
    f.write_text(PY)
    code = cli.main(["patch", "--file", str(f), "--before", "# X", "--after", "# Y"])
    assert code == 3
    assert f.read_text() == PY            # not written — verification failed


def test_allow_broken_writes_despite_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    return (((\n    "))
    f = tmp_path / "m.py"
    f.write_text(PY)
    code = cli.main(["patch", "--file", str(f), "--before", "# X", "--after", "# Y", "--allow-broken"])
    assert code == 0
    assert "return (((" in f.read_text()


def test_json_output_contract(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    y = 2\n    "))
    f = tmp_path / "m.py"
    f.write_text(PY)
    code = cli.main(["patch", "--file", str(f), "--before", "# X", "--after", "# Y", "--json", "--dry-run"])
    assert code == 0
    record = json.loads(capsys.readouterr().out)
    assert record["bounded"] is True
    assert record["parses"] is True
    assert record["wrote"] is False
    assert "frozen_prefix_chars" in record and "unified_diff" in record


def test_text_mode_prints_result(monkeypatch, capsys):
    monkeypatch.setattr("vibe_fim.cli.codestral_fim", _fake_backend("\n    y = 2\n    "))
    code = cli.main(["patch", "--text", PY, "--before", "# X", "--after", "# Y", "--name", "m.py"])
    assert code == 0
    assert "y = 2" in capsys.readouterr().out


def test_no_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.main([])
