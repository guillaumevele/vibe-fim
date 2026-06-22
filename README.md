# vibe-fim

**Bounded AI code edits via Codestral fill-in-the-middle.** Rewrite only the
region between two anchors — the rest of the file comes out *byte-for-byte
identical*, by construction.

[![CI](https://github.com/guillaumevele/vibe-fim/actions/workflows/ci.yml/badge.svg)](https://github.com/guillaumevele/vibe-fim/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![deps](https://img.shields.io/badge/dependencies-none-brightgreen)

---

## Why this exists

Most coding agents edit by **regenerating a block** and pasting it back. The
model re-emits your whole function (or file) token by token, so it can *silently*
change things you never asked it to touch — reflow a comment, drop a trailing
newline, "fix" an unrelated line, normalise quotes.

Fill-in-the-middle (FIM) makes that **structurally impossible**. You hand the
model a verbatim *prefix* and *suffix*; it only writes the *middle*. So the edit
is provably confined to the region between two anchors. Everything outside is
never sent to be rewritten — it cannot drift.

`vibe-fim` wraps Mistral's **Codestral FIM** in a one-call surgical-patch
primitive and *verifies* the guarantee on every edit: the output must start with
the exact prefix and end with the exact suffix, or the patch is rejected.

> This is the one thing competing agent CLIs can't express: none of Claude Code,
> Codex, or vanilla `vibe` has a FIM patch primitive. Codestral does. `vibe-fim`
> turns it into a bounded edit you can trust.

## Install

```bash
pip install vibe-fim
export MISTRAL_API_KEY=...   # from https://console.mistral.ai
```

Zero runtime dependencies (stdlib `urllib` only).

## CLI

```bash
# Rewrite only what's between the two anchors; show the diff without writing.
vibe-fim patch --file units.py \
  --before "def to_celsius(f):" \
  --after  $'\n\ndef to_fahrenheit' \
  --hint   "round the result to 1 decimal place" \
  --dry-run
```

The scope report goes to stderr:

```
[BOUNDED] edit confined between anchors.
  frozen prefix: 142 chars (unchanged)
  frozen suffix: 98 chars (unchanged)
  region: 31 chars -> 39 chars
```

Drop `--dry-run` to write the file in place.

## Library

```python
from vibe_fim import patch_region

result = patch_region(
    source_code,
    before="def to_celsius(f):",   # frozen prefix ends right after this
    after="\n\ndef to_fahrenheit", # frozen suffix starts right at this
    hint="round the result to 1 decimal place",
)

assert result.bounded                       # verified, not assumed
assert source_code.startswith(result.prefix)  # prefix is byte-identical
new_source = result.text
```

The completion backend is injectable, so the bounded-edit logic is fully
testable with no network:

```python
patch_region(src, before, after, fim_fn=lambda prefix, suffix: "    return 42")
```

Anchors must be **unique** in the file. A missing or ambiguous anchor raises
`FimError` instead of guessing — surgical edits should never be applied to the
wrong place.

### Robustness: no duplicated lines

FIM models sometimes re-emit the first lines of the suffix (or the last lines of
the prefix) inside the middle. `patch_region` detects and strips that overlap —
but **only at a line boundary**, so a coincidental shared `}` or `)` is never
removed. When it happens, `result.trimmed_overlap` reports how many characters
were de-duplicated.

## The guarantee, benchmarked

`examples/bench.py` runs the same edit ("round `to_celsius` to 1 decimal") two
ways and counts changed lines *outside the target function*:

| approach | collateral changes outside target | guaranteed? |
|---|---|---|
| **surgical FIM** (`vibe-fim`) | **0** | yes — by construction |
| block regeneration (chat returns the file) | drifts (e.g. dropped the file's trailing newline in our run) | no |

The block-regen baseline produced the *correct* body — and still silently
altered the file outside the edit. FIM can't, because the prefix and suffix are
never regenerated.

```bash
MISTRAL_API_KEY=... python examples/bench.py
```

## How it works

1. Locate the unique `before` and `after` anchors.
2. `prefix = text[:end_of_before]`, `suffix = text[start_of_after:]`.
3. Send `(prefix, suffix)` to Codestral FIM → `middle`.
4. `new_text = prefix + middle + suffix`.
5. **Verify** `new_text.startswith(prefix) and new_text.endswith(suffix)` —
   else raise. The bytes outside the region are identical because they're the
   same object slices, concatenated unchanged.

## License

MIT © Guillaume Vele
