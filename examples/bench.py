#!/usr/bin/env python3
"""Reproducible mini-benchmark: surgical FIM vs. block regeneration.

Same file, same edit request ("round to_celsius to 1 decimal"). We measure how
many lines OUTSIDE the target function change.

- Surgical FIM (vibe-fim): the prefix and suffix are sent verbatim and never
  rewritten, so collateral change is structurally impossible — 0, guaranteed.
- Block regeneration (a normal chat call that returns the whole file): the
  correct body is produced, but the rest of the file is re-emitted token by
  token, so silent drift can creep in (reformatting, dropped trailing newline,
  "helpful" comment edits). No guarantee.

Run with MISTRAL_API_KEY set:  python examples/bench.py
"""
from __future__ import annotations

import difflib
import json
import os
import urllib.request

from vibe_fim import patch_region

FILE = '''\
# units.py — temperature helpers
# version = "1.4.0"

TABLE_HEADER = 'F | C'   # legacy single-quote style, do not change

def to_celsius(f):
    return (f - 32) * 5 / 9


def to_fahrenheit(c):
    return c * 9 / 5 + 32


def format_row(f):
    c = to_celsius(f)
    return TABLE_HEADER + '\\n' + str(f) + ' | ' + str(c)
'''

EDIT = "make to_celsius round the result to 1 decimal place"
TARGET_LINES = {"def to_celsius(f):", "    return (f - 32) * 5 / 9"}


def changed_lines_outside_target(original: str, edited: str) -> int:
    """Count changed lines that are not part of the target function body."""
    diff = difflib.unified_diff(original.splitlines(), edited.splitlines(), n=0)
    count = 0
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            body = line[1:]
            if body.strip() and not any(t in body for t in ("to_celsius", "(f - 32)", "round(")):
                count += 1
    return count


def block_regenerate(file_text: str, edit: str) -> str:
    """Baseline: ask a chat model to return the whole updated file."""
    body = json.dumps({
        "model": "codestral-latest",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Return ONLY the complete updated file, no markdown fences."},
            {"role": "user", "content": f"Edit request: {edit}.\n\nFile:\n{file_text}"},
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main() -> None:
    # Surgical FIM: anchors bound the edit to to_celsius's body.
    surgical = patch_region(
        FILE, "def to_celsius(f):", "\n\ndef to_fahrenheit",
        hint="round the result to 1 decimal place",
    )
    surgical_outside = 0 if surgical.bounded else 99  # bounded => 0 by construction

    # Block regen baseline.
    regen = block_regenerate(FILE, EDIT)
    if not regen.endswith("\n"):
        regen += ""  # keep raw; trailing-newline drift is exactly what we measure
    regen_outside = changed_lines_outside_target(FILE, regen)

    print("=== vibe-fim mini-bench: edit to_celsius, measure collateral ===\n")
    print(f"surgical FIM   : prefix+suffix byte-identical = {surgical.bounded}; "
          f"lines changed outside target = {surgical_outside}")
    print(surgical.scope_report())
    print()
    print(f"block regen    : lines changed outside target = {regen_outside}")
    print(f"               : trailing newline preserved = {regen.endswith(chr(10))}")


if __name__ == "__main__":
    main()
