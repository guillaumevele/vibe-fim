"""Tests for transactional multi-region FIM (patch_regions). Backend injected."""
from __future__ import annotations

import pytest

from vibe_fim import FimError, patch_regions

FILE = '''\
import os

# A: greet
def greet(name):
    # GA
    return "hi"
    # GE

# B: add
def add(a, b):
    # AA
    return 0
    # AE

TRAILER = 1
'''


def _seq(*middles):
    it = iter(middles)
    return lambda prefix, suffix: next(it)


def test_two_regions_applied_frozen_segments_byte_identical():
    result = patch_regions(
        FILE,
        [{"before": "# GA", "after": "# GE"},
         {"before": "# AA", "after": "# AE"}],
        fim_fn=_seq('\n    return f"hi {name}"\n    ', '\n    return a + b\n    '),
        filename="m.py",
    )
    assert result.region_count == 2
    assert result.bounded is True
    assert result.parses is True
    # both edits landed
    assert 'return f"hi {name}"' in result.text
    assert "return a + b" in result.text
    # everything between/around is byte-identical: imports, the other defs' headers,
    # the trailer — all survive verbatim.
    assert result.text.startswith("import os\n")
    assert "def greet(name):" in result.text and "def add(a, b):" in result.text
    assert result.text.endswith("TRAILER = 1\n")
    # exact reconstruction holds
    expected = result.frozen[0]
    for i, m in enumerate(result.new_middles):
        expected += m + result.frozen[i + 1]
    assert result.text == expected


def test_overlapping_edits_rejected_nothing_applied():
    with pytest.raises(FimError, match="overlap"):
        patch_regions(FILE,
                      [{"before": "# GA", "after": "# AE"},
                       {"before": "# GE", "after": "# AA"}],
                      fim_fn=_seq("x", "y"))


def test_out_of_order_edits_rejected():
    with pytest.raises(FimError, match="file order"):
        patch_regions(FILE,
                      [{"before": "# AA", "after": "# AE"},
                       {"before": "# GA", "after": "# GE"}],
                      fim_fn=_seq("x", "y"))


def test_missing_anchor_rejects_whole_transaction():
    with pytest.raises(FimError, match="not found"):
        patch_regions(FILE,
                      [{"before": "# GA", "after": "# GE"},
                       {"before": "# NOPE", "after": "# AE"}],
                      fim_fn=_seq("x", "y"))


def test_parse_failure_flags_the_whole_unit():
    result = patch_regions(
        FILE,
        [{"before": "# GA", "after": "# GE"},
         {"before": "# AA", "after": "# AE"}],
        fim_fn=_seq('\n    return 1\n    ', '\n    return (((\n    '),  # 2nd breaks syntax
        filename="m.py",
    )
    assert result.parses is False
    assert "DOES NOT PARSE" in result.scope_report()


def test_empty_edits_raises():
    with pytest.raises(FimError, match="no edits"):
        patch_regions(FILE, [], fim_fn=_seq())


def test_echo_dedup_across_regions_counted():
    # second region echoes the start of its suffix on a full line
    result = patch_regions(
        FILE,
        [{"before": "# GA", "after": "# GE"},
         {"before": "# AA", "after": "# AE"}],
        fim_fn=_seq('\n    return 1\n    ', '\n    return 2\n    # AE\n'),
        filename="m.py",
    )
    assert result.text.count("# AE") == 1
    assert result.trimmed_overlap > 0
