#!/usr/bin/env python3
"""Reproducible mini-benchmark: surgical FIM vs. block regeneration.

Same file, same edit ("round to_celsius to 1 decimal"). We count how many lines
change OUTSIDE the target function, using ONE newline-aware counter for both
approaches — so the comparison is a real measurement, not a postcondition
restating itself.

By default this REPLAYS a committed transcript captured from Codestral
(examples/fixtures/), so it runs offline with no API key and any reviewer gets
the same numbers. Pass --live to re-capture from the Mistral API instead.

  python examples/bench.py            # offline replay of the committed fixture
  MISTRAL_API_KEY=... python examples/bench.py --live
"""
from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path

from vibe_fim import patch_region

FIX = Path(__file__).parent / "fixtures"
ANCHOR_BEFORE = "def to_celsius(f):"
ANCHOR_AFTER = "\n\ndef to_fahrenheit"
EDIT = "make to_celsius round the result to 1 decimal place"
# Lines that ARE the target — changes here are expected, not collateral.
TARGET_TOKENS = ("to_celsius", "(f - 32)", "round(")


def changed_lines_outside_target(original: str, edited: str) -> int:
    """Newline-aware: keepends=True so a dropped trailing newline is counted, not
    swallowed. Counts changed lines that are not part of the target function."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        edited.splitlines(keepends=True),
        n=0,
    )
    count = 0
    for line in diff:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            body = line[1:]
            if not any(tok in body for tok in TARGET_TOKENS):
                count += 1
    return count


def load_offline() -> tuple[str, str, str]:
    before = (FIX / "units_before.py").read_text()
    regen = (FIX / "regen_after.py").read_text()
    fim_middle = json.loads((FIX / "transcript.json").read_text())["fim_middle"]
    return before, regen, fim_middle


def load_live() -> tuple[str, str, str]:
    import urllib.request
    before = (FIX / "units_before.py").read_text()
    key = os.environ["MISTRAL_API_KEY"]

    def chat(messages):
        body = json.dumps({"model": "codestral-latest", "temperature": 0,
                           "messages": messages}).encode()
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    regen = chat([
        {"role": "system", "content": "Return ONLY the complete updated file, no fences."},
        {"role": "user", "content": f"Edit request: {EDIT}.\n\nFile:\n{before}"},
    ])
    # For the surgical path, capture a real FIM middle for the target region.
    fim_middle = patch_region(before, ANCHOR_BEFORE, ANCHOR_AFTER,
                              hint="round the result to 1 decimal place",
                              filename="units.py").new_middle
    return before, regen, fim_middle


def main() -> None:
    live = "--live" in sys.argv
    before, regen, fim_middle = load_live() if live else load_offline()

    surgical = patch_region(before, ANCHOR_BEFORE, ANCHOR_AFTER,
                            fim_fn=lambda p, s: fim_middle, filename="units.py")

    surgical_outside = changed_lines_outside_target(before, surgical.text)
    regen_outside = changed_lines_outside_target(before, regen)

    print(f"=== vibe-fim mini-bench ({'live' if live else 'offline replay'}) ===\n")
    print("THE GUARANTEE (structural, by construction):")
    print(f"  surgical FIM — prefix+suffix are the original slices; "
          f"bounded={surgical.bounded}, parses={surgical.parses}\n")
    print("THE DRIFT (measured, same newline-aware counter for both):")
    print(f"  surgical FIM      : {surgical_outside} lines changed outside target")
    print(f"  block regeneration: {regen_outside} lines changed outside target")
    print(f"  (block regen trailing newline preserved: {regen.endswith(chr(10))})")
    if not live:
        print("\n  source: examples/fixtures/transcript.json (committed Codestral capture)")


if __name__ == "__main__":
    main()
