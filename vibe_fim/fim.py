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
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise FimError(f"Unexpected FIM response shape: {payload}") from exc
    # A completion cut off at the token cap is an INCOMPLETE middle — the bounded
    # edit would be silently half-written. Fail loudly rather than ship it.
    if choice.get("finish_reason") == "length":
        raise FimError(
            f"completion truncated at max_tokens={max_tokens}; the middle is "
            "incomplete — raise max_tokens and retry."
        )
    if content is None or content == "":
        raise FimError("Codestral FIM returned empty content (no middle).")
    return content


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
    prefix: str          # frozen prefix — a byte-identical slice of the original (invariant)
    suffix: str          # frozen suffix — a byte-identical slice of the original (invariant)
    old_middle: str      # what was between the anchors before
    new_middle: str      # what the model wrote (after echo de-duplication)
    # Structural invariant: text == prefix + new_middle + suffix. True by
    # construction; the assertion guards against a future refactor breaking it.
    # The verification that can actually FAIL is `parses` and the truncation /
    # echo checks upstream — not this field.
    bounded: bool
    trimmed_overlap: int = 0  # chars of suffix/prefix the model re-emitted, then stripped
    parses: bool | None = None  # did the patched file still parse? None = not checked

    @property
    def changed_chars(self) -> int:
        return len(self.new_middle) + len(self.old_middle)

    def scope_report(self) -> str:
        tag = "BOUNDED" if self.bounded else "SCOPE VIOLATION"
        if self.parses is True:
            tag += " + PARSES"
        elif self.parses is False:
            tag += " + DOES NOT PARSE"
        report = (
            f"[{tag}] edit confined between anchors.\n"
            f"  frozen prefix: {len(self.prefix)} chars (unchanged)\n"
            f"  frozen suffix: {len(self.suffix)} chars (unchanged)\n"
            f"  region: {len(self.old_middle)} chars -> {len(self.new_middle)} chars"
        )
        if self.trimmed_overlap:
            report += f"\n  de-duplicated {self.trimmed_overlap} re-emitted chars at a boundary"
        return report


# Distinguish a real re-emission from coincidental boundary formatting:
#   - a lone boundary newline or indentation (`\n`, `\n    `) strips to nothing
#     → coincidence, keep it;
#   - a single shared `}` or `);` on one line is short and ambiguous → keep it;
#   - a run of >= _MIN_ECHO_CHARS non-whitespace on one line, OR a re-emitted
#     COMPLETE line (contains a newline and has real content), is a genuine echo
#     → strip it. The whole-line rule catches short markers like `# x` / `});`
#     that fall under the char threshold but are unambiguous when re-emitted whole.
_MIN_ECHO_CHARS = 4


def _is_echo(segment: str) -> bool:
    stripped = segment.strip()
    if not stripped:
        return False  # only whitespace / a boundary newline — coincidental
    return len(stripped) >= _MIN_ECHO_CHARS or "\n" in segment


def _trim_boundary_overlap(middle: str, prefix: str, suffix: str) -> tuple[str, int]:
    """FIM models sometimes re-emit the start of the suffix (or the end of the
    prefix) inside the middle, silently duplicating code — on the SAME line, not
    just across a line break. Strip such a re-emission when it is a real echo
    (a whole line, or a run of at least `_MIN_ECHO_CHARS`), while leaving a
    coincidental shared `}` or `)` untouched."""
    trimmed = 0
    # Tail of middle that duplicates the head of the suffix.
    for k in range(min(len(middle), len(suffix)), 0, -1):
        if middle[-k:] == suffix[:k]:
            if _is_echo(middle[-k:]):
                middle = middle[:-k]
                trimmed += k
            break
    # Head of middle that duplicates the tail of the prefix.
    for k in range(min(len(middle), len(prefix)), 0, -1):
        if middle[:k] == prefix[-k:]:
            if _is_echo(middle[:k]):
                middle = middle[k:]
                trimmed += k
            break
    return middle, trimmed


def _check_parses(text: str, filename: str | None) -> bool | None:
    """A real, can-FAIL verification: does the patched file still parse?

    Returns True/False for languages we can check with the stdlib (Python via
    `ast`), or None when the language is unknown — byte-bounded edits are not the
    same as syntactically-valid ones, and this is what makes "verified" mean
    something a broken backend can actually trip."""
    if not filename:
        return None
    if filename.endswith((".py", ".pyi")):
        import ast
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False
    return None


# Comment syntax by file extension. The hint is sent to the model to steer the
# completion; it is NEVER written to the file. We pick the right syntax so the
# hint does not look like code in the wrong language — and emit no comment at all
# when we cannot be sure, rather than guess wrong.
_HASH = "#"
_SLASH = "//"
_COMMENT_BY_EXT = {
    **dict.fromkeys((".py", ".pyi", ".rb", ".sh", ".bash", ".zsh", ".yaml",
                     ".yml", ".toml", ".cfg", ".ini", ".r", ".pl", ".tf"), _HASH),
    **dict.fromkeys((".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".h",
                     ".cpp", ".cc", ".hpp", ".java", ".kt", ".swift", ".scala",
                     ".cs", ".php", ".m", ".mm", ".metal", ".dart", ".proto"), _SLASH),
    **dict.fromkeys((".sql", ".lua", ".hs", ".elm", ".adb"), "--"),
}
_BLOCK_BY_EXT = {
    ".css": ("/*", "*/"), ".scss": ("/*", "*/"),
    ".html": ("<!--", "-->"), ".xml": ("<!--", "-->"), ".svg": ("<!--", "-->"),
}


def _comment_line(filename: str | None, prefix: str, hint: str) -> str | None:
    """Render `hint` as a comment line in the file's language, or None when no
    safe comment style is known (then the hint is simply not sent)."""
    ext = ""
    if filename:
        dot = filename.rfind(".")
        ext = filename[dot:].lower() if dot != -1 else ""
    if ext in _BLOCK_BY_EXT:
        open_, close_ = _BLOCK_BY_EXT[ext]
        return f"{open_} {hint} {close_}"
    if ext in _COMMENT_BY_EXT:
        return f"{_COMMENT_BY_EXT[ext]} {hint}"
    if not ext:  # no filename: fall back to a light heuristic on the prefix
        if "//" in prefix or any(t in prefix for t in ("func ", "function ", "const ", "let ")):
            return f"{_SLASH} {hint}"
        if any(t in prefix for t in ("def ", "import ", "self.")):
            return f"{_HASH} {hint}"
    return None  # unknown language → no comment rather than a wrong one


def patch_region(
    text: str,
    before: str,
    after: str,
    *,
    hint: str = "",
    fim_fn: FimFn | None = None,
    filename: str | None = None,
    verify: bool = True,
) -> PatchResult:
    """Replace the region strictly between `before` and `after` with a FIM
    completion, guaranteeing everything outside stays byte-identical.

    - `before`:   a unique anchor; the frozen prefix ends right after it.
    - `after`:    a unique anchor; the frozen suffix starts right at it.
    - `hint`:     optional steer, sent to the model as a trailing comment but NOT
                  written to the file.
    - `filename`: used only to pick a parser for the post-patch parse check.
    - `verify`:   when True, run the parse check and record `result.parses`.
    """
    _, before_end = _locate_unique(text, before, "before")
    after_start, _ = _locate_unique(text, after, "after")
    if after_start < before_end:
        raise FimError("`after` anchor occurs before `before` anchor.")

    prefix = text[:before_end]
    suffix = text[after_start:]
    old_middle = text[before_end:after_start]

    fim_fn = fim_fn or codestral_fim
    fim_prefix = prefix
    if hint:
        comment = _comment_line(filename, prefix, hint)
        if comment:
            fim_prefix = f"{prefix}\n{comment}\n"
    raw_middle = fim_fn(fim_prefix, suffix)

    # Guard against the model re-emitting whole lines of the frozen prefix/suffix.
    # Use `fim_prefix` (prefix + hint comment) on the prefix side: a model that
    # echoes the injected hint comment at the head of the middle is detected and
    # stripped, so the hint never leaks into the written file.
    new_middle, trimmed = _trim_boundary_overlap(raw_middle, fim_prefix, suffix)

    new_text = prefix + new_middle + suffix
    # Structural invariant (true by construction); kept as a cheap guard.
    bounded = new_text.startswith(prefix) and new_text.endswith(suffix)
    if not bounded:
        raise FimError(
            "Backend did not preserve the prefix/suffix (FIM contract violated)."
        )
    parses = _check_parses(new_text, filename) if verify else None
    return PatchResult(
        text=new_text,
        prefix=prefix,
        suffix=suffix,
        old_middle=old_middle,
        new_middle=new_middle,
        bounded=bounded,
        trimmed_overlap=trimmed,
        parses=parses,
    )
