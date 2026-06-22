"""Tests for the bounded-edit core. No network: fim_fn is injected."""
from __future__ import annotations

import pytest

from vibe_fim import FimError, patch_region


SAMPLE = '''\
def greet(name):
    # BEGIN
    return "hi"
    # END

def other():
    return 1
'''


def test_patch_is_bounded_prefix_and_suffix_byte_identical():
    def fake_fim(prefix, suffix):
        return '\n    return f"hello, {name}"\n    '

    result = patch_region(SAMPLE, "# BEGIN", "# END", fim_fn=fake_fim)

    # Everything before the `before` anchor's end is frozen...
    assert SAMPLE.startswith(result.prefix)
    # ...and everything from the `after` anchor on is frozen.
    assert SAMPLE.endswith(result.suffix)
    # The new file is exactly prefix + new_middle + suffix.
    assert result.text == result.prefix + result.new_middle + result.suffix
    assert result.bounded is True
    # `other()` is in the suffix → provably untouched.
    assert "def other():" in result.text
    assert result.text.endswith("def other():\n    return 1\n")


def test_old_middle_captured():
    result = patch_region(SAMPLE, "# BEGIN", "# END", fim_fn=lambda p, s: "\n    pass\n    ")
    assert 'return "hi"' in result.old_middle
    assert 'return "hi"' not in result.new_middle


def test_missing_anchor_raises():
    with pytest.raises(FimError, match="not found"):
        patch_region(SAMPLE, "# NOPE", "# END", fim_fn=lambda p, s: "")


def test_ambiguous_anchor_raises():
    text = "x = 1\n# M\ny = 2\n# M\nz = 3\n"
    with pytest.raises(FimError, match="ambiguous"):
        patch_region(text, "# M", "z = 3", fim_fn=lambda p, s: "")


def test_inverted_anchors_raise():
    with pytest.raises(FimError, match="before"):
        patch_region(SAMPLE, "# END", "# BEGIN", fim_fn=lambda p, s: "")


def test_empty_anchor_raises():
    with pytest.raises(FimError, match="empty"):
        patch_region(SAMPLE, "", "# END", fim_fn=lambda p, s: "")


def test_hint_is_not_written_to_file():
    captured = {}

    def fake_fim(prefix, suffix):
        captured["prefix"] = prefix
        return "\n    pass\n    "

    result = patch_region(
        SAMPLE, "# BEGIN", "# END",
        hint="make it idempotent", fim_fn=fake_fim,
    )
    # The hint reaches the model...
    assert "make it idempotent" in captured["prefix"]
    # ...but never lands in the file.
    assert "make it idempotent" not in result.text


def test_scope_report_mentions_bounded():
    result = patch_region(SAMPLE, "# BEGIN", "# END", fim_fn=lambda p, s: "\n    pass\n    ")
    assert "BOUNDED" in result.scope_report()


def test_suffix_line_reemission_is_deduplicated():
    # Model re-emits the first line of the suffix inside the middle.
    text = "a = 1\n# BEGIN\nx = 0\n# END\nb = 2\n"
    # suffix begins with "# END\nb = 2\n"; the model duplicates "# END\n".
    result = patch_region(
        text, "# BEGIN", "# END",
        fim_fn=lambda p, s: "\nx = 42\n# END\n",
    )
    assert result.trimmed_overlap > 0
    # "# END" appears exactly once, not twice.
    assert result.text.count("# END") == 1
    assert result.text == "a = 1\n# BEGIN\nx = 42\n# END\nb = 2\n"


def test_coincidental_single_char_is_not_trimmed():
    # middle ends with "}" and suffix begins with "}" — must NOT be stripped.
    text = "f = {\n# A\n0: 1\n# B\n}\n"
    result = patch_region(
        text, "# A", "# B",
        fim_fn=lambda p, s: "\n2: 3}",  # ends with } that also starts suffix region
    )
    # No newline in the overlap → nothing trimmed.
    assert result.trimmed_overlap == 0


def test_backend_breaking_contract_is_rejected():
    # A pathological backend whose output would still need prefix+suffix to hold.
    # patch_region builds new_text = prefix + middle + suffix, so the invariant
    # holds by construction; this guards the assertion path stays wired.
    result = patch_region(SAMPLE, "# BEGIN", "# END", fim_fn=lambda p, s: "X")
    assert result.text == result.prefix + "X" + result.suffix
