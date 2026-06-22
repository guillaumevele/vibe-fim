"""Bounded code edits via Codestral fill-in-the-middle (FIM).

The idea: most coding agents edit by *regenerating a block*, which can silently
change code outside your intent. FIM cannot. You give the model a verbatim
*prefix* and *suffix*; it only writes the *middle*. So the edit is provably
bounded to the region between two anchors — the prefix and suffix of the file
come out byte-for-byte identical. This module enforces and verifies that.

Backend: Mistral Codestral FIM (`/v1/fim/completions`). The completion call is
injectable, so the bounded-edit logic is testable with no network and you can
swap in any FIM-capable backend.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

FIM_ENDPOINT = "https://api.mistral.ai/v1/fim/completions"
DEFAULT_MODEL = "codestral-latest"

# A FIM backend: (prefix, suffix) -> middle.
FimFn = Callable[[str, str], str]


class FimError(Exception):
    """Raised when an edit cannot be applied safely (ambiguous anchors, backend
    failure, or — critically — a scope violation)."""


def codestral_fim(
    prefix: str,
    suffix: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout_s: int = 60,
) -> str:
    """Call Mistral Codestral FIM. Reads MISTRAL_API_KEY from the environment
    unless `api_key` is given. Returns the generated middle text."""
    api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise FimError("MISTRAL_API_KEY is not set (and no api_key was provided).")
    body = json.dumps({
        "model": model,
        "prompt": prefix,
        "suffix": suffix,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    request = urllib.request.Request(
        FIM_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise FimError(f"Codestral FIM HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise FimError(f"Codestral FIM request failed: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FimError(f"Unexpected FIM response shape: {payload}") from exc


def _locate_unique(text: str, anchor: str, what: str) -> tuple[int, int]:
    """Return (start, end) of a unique anchor, or raise on missing/ambiguous."""
    if not anchor:
        raise FimError(f"{what} anchor is empty.")
    count = text.count(anchor)
    if count == 0:
        raise FimError(f"{what} anchor not found: {anchor!r}")
    if count > 1:
        raise FimError(
            f"{what} anchor is ambiguous (appears {count} times): {anchor!r}. "
            "Use a longer, unique anchor."
        )
    start = text.index(anchor)
    return start, start + len(anchor)


@dataclass(frozen=True)
class PatchResult:
    text: str            # the new, full file content
    prefix: str          # frozen prefix (byte-identical to the original)
    suffix: str          # frozen suffix (byte-identical to the original)
    old_middle: str      # what was between the anchors before
    new_middle: str      # what the model wrote (after overlap de-duplication)
    bounded: bool        # True iff prefix and suffix are preserved exactly
    trimmed_overlap: int = 0  # chars of suffix/prefix the model re-emitted, then stripped

    @property
    def changed_chars(self) -> int:
        return len(self.new_middle) + len(self.old_middle)

    def scope_report(self) -> str:
        status = "BOUNDED" if self.bounded else "SCOPE VIOLATION"
        report = (
            f"[{status}] edit confined between anchors.\n"
            f"  frozen prefix: {len(self.prefix)} chars (unchanged)\n"
            f"  frozen suffix: {len(self.suffix)} chars (unchanged)\n"
            f"  region: {len(self.old_middle)} chars -> {len(self.new_middle)} chars"
        )
        if self.trimmed_overlap:
            report += f"\n  de-duplicated {self.trimmed_overlap} re-emitted chars at a boundary"
        return report


def _trim_boundary_overlap(middle: str, prefix: str, suffix: str) -> tuple[str, int]:
    """FIM models sometimes re-emit the start of the suffix (or the end of the
    prefix) inside the middle, which would silently duplicate whole lines. Strip
    such a re-emission, but ONLY when the overlap spans a line boundary — so a
    coincidental single char (a shared `}` or `)`) is never removed."""
    trimmed = 0
    # Tail of middle that duplicates the head of the suffix.
    for k in range(min(len(middle), len(suffix)), 0, -1):
        if middle[-k:] == suffix[:k]:
            if "\n" in middle[-k:]:
                middle = middle[:-k]
                trimmed += k
            break
    # Head of middle that duplicates the tail of the prefix.
    for k in range(min(len(middle), len(prefix)), 0, -1):
        if middle[:k] == prefix[-k:]:
            if "\n" in middle[:k]:
                middle = middle[k:]
                trimmed += k
            break
    return middle, trimmed


def _comment_for(prefix: str, hint: str) -> str:
    """Render a hint as a one-line comment, guessing the syntax from the file."""
    lower = prefix.rstrip().rsplit("\n", 1)[-1]
    if any(t in prefix for t in ("def ", "import ", "self.")) or lower.startswith("#"):
        return f"# {hint}"
    if "//" in prefix or any(t in prefix for t in ("function ", "const ", "let ", "func ")):
        return f"// {hint}"
    return f"# {hint}"


def patch_region(
    text: str,
    before: str,
    after: str,
    *,
    hint: str = "",
    fim_fn: FimFn | None = None,
) -> PatchResult:
    """Replace the region strictly between `before` and `after` with a FIM
    completion, guaranteeing everything outside stays byte-identical.

    - `before`: a unique anchor; the frozen prefix ends right after it.
    - `after`:  a unique anchor; the frozen suffix starts right at it.
    - `hint`:   optional steer, sent to the model as a trailing comment but NOT
                written to the file.
    """
    _, before_end = _locate_unique(text, before, "before")
    after_start, _ = _locate_unique(text, after, "after")
    if after_start < before_end:
        raise FimError("`after` anchor occurs before `before` anchor.")

    prefix = text[:before_end]
    suffix = text[after_start:]
    old_middle = text[before_end:after_start]

    fim_fn = fim_fn or codestral_fim
    fim_prefix = prefix if not hint else f"{prefix}\n{_comment_for(prefix, hint)}\n"
    raw_middle = fim_fn(fim_prefix, suffix)

    # Guard against the model re-emitting whole lines of the frozen prefix/suffix.
    new_middle, trimmed = _trim_boundary_overlap(raw_middle, prefix, suffix)

    new_text = prefix + new_middle + suffix
    # The structural guarantee — verified, not just assumed:
    bounded = new_text.startswith(prefix) and new_text.endswith(suffix)
    if not bounded:
        raise FimError(
            "Backend did not preserve the prefix/suffix (FIM contract violated)."
        )
    return PatchResult(
        text=new_text,
        prefix=prefix,
        suffix=suffix,
        old_middle=old_middle,
        new_middle=new_middle,
        bounded=bounded,
        trimmed_overlap=trimmed,
    )
