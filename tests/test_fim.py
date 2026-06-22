"""Tests for the bounded-edit core. No network: fim_fn is injected, and the
HTTP layer is exercised with a fake urlopen (never a real request)."""
from __future__ import annotations

import contextlib
import io
import json
import urllib.error
import urllib.request

import pytest

from vibe_fim.fim import codestral_fim

from vibe_fim import FimError, patch_region


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@contextlib.contextmanager
def _patched_urlopen(fake):
    original = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        yield
    finally:
        urllib.request.urlopen = original


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


def test_same_line_echo_is_deduplicated():
    # The model re-emits the head of the suffix on the SAME line (no newline).
    # Regression: the old newline-only guard let this through, duplicating code
    # inside the region while still reporting bounded=True.
    text = "head\n#A anchor line\nMID\n#B trailer xyz tail"
    result = patch_region(
        text, "#A anchor line", "#B trailer xyz",
        fim_fn=lambda p, s: "\nnew body\n" + s[:14],  # echoes "#B trailer xyz"
    )
    assert result.text.count("#B trailer xyz") == 1
    assert result.trimmed_overlap == 14


def test_boundary_newline_is_not_trimmed():
    # The middle's trailing newline coincides with the suffix's leading newline.
    # That is coincidental formatting, not a re-emitted line — it must be kept,
    # or every block-bodied edit would lose a newline at the boundary.
    text = "def f():\n    # X\n    old\n\ndef g():\n    pass\n"
    result = patch_region(text, "# X", "\n\ndef g():", fim_fn=lambda p, s: "\n    return 1\n")
    assert result.trimmed_overlap == 0
    assert result.new_middle == "\n    return 1\n"


def test_parse_gate_passes_on_valid_python():
    result = patch_region(
        "def f():\n    # X\n    pass\n    # Y\n    return 1\n",
        "# X", "# Y", filename="m.py",
        fim_fn=lambda p, s: "\n    y = 2\n    ",
    )
    assert result.parses is True
    assert "PARSES" in result.scope_report()


def test_parse_gate_fails_on_broken_python():
    result = patch_region(
        "def f():\n    # X\n    pass\n    # Y\n    return 1\n",
        "# X", "# Y", filename="m.py",
        fim_fn=lambda p, s: "\n    return (((\n    ",
    )
    assert result.parses is False
    assert "DOES NOT PARSE" in result.scope_report()


def test_parse_gate_is_none_for_unknown_language():
    result = patch_region(SAMPLE, "# BEGIN", "# END", fim_fn=lambda p, s: "\n  pass\n  ")
    assert result.parses is None  # no filename → no parser


def test_truncated_completion_raises():
    captured = []

    def fake_urlopen(req, timeout=None):
        body = json.dumps({"choices": [{"message": {"content": "half"},
                                        "finish_reason": "length"}]}).encode()
        return _FakeResp(body)

    with _patched_urlopen(fake_urlopen):
        with pytest.raises(FimError, match="truncated"):
            from vibe_fim.fim import codestral_fim
            codestral_fim("p", "s", api_key="k")


def test_none_content_raises():
    def fake_urlopen(req, timeout=None):
        body = json.dumps({"choices": [{"message": {"content": None},
                                        "finish_reason": "stop"}]}).encode()
        return _FakeResp(body)

    with _patched_urlopen(fake_urlopen):
        with pytest.raises(FimError, match="empty completion|empty content"):
            codestral_fim("p", "s", api_key="k")


def test_http_happy_path():
    def fake_urlopen(req, timeout=None):
        body = json.dumps({"choices": [{"message": {"content": "MIDDLE"},
                                        "finish_reason": "stop"}]}).encode()
        return _FakeResp(body)

    with _patched_urlopen(fake_urlopen):
        assert codestral_fim("p", "s", api_key="k") == "MIDDLE"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(FimError, match="MISTRAL_API_KEY"):
        codestral_fim("p", "s")


def test_malformed_json_raises():
    def fake_urlopen(req, timeout=None):
        return _FakeResp(b"this is not json")

    with _patched_urlopen(fake_urlopen):
        with pytest.raises(FimError, match="request failed"):
            codestral_fim("p", "s", api_key="k")


def test_http_error_surfaces_status():
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.mistral.ai", 429, "Too Many Requests", {},
            io.BytesIO(b'{"message":"rate limit"}'))

    with _patched_urlopen(fake_urlopen):
        with pytest.raises(FimError, match="HTTP 429"):
            codestral_fim("p", "s", api_key="k")


def test_timeout_raises():
    def fake_urlopen(req, timeout=None):
        raise TimeoutError("timed out")

    with _patched_urlopen(fake_urlopen):
        with pytest.raises(FimError, match="request failed"):
            codestral_fim("p", "s", api_key="k")
