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
primitive. The bytes outside the region are the original slices, concatenated
unchanged — so *bounded scope* is structural. On top of that it runs the checks
that can actually **fail** (below), so a bad completion is caught, not shipped.

> No agent CLI exposes a *verified bounded-edit* primitive. Claude Code and Codex
> edit by regenerating blocks; plain `vibe` has no FIM patch op. Codestral's FIM
> is what makes a provably-bounded edit possible, and `vibe-fim` turns it into one
> an agent can call and check.

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
[BOUNDED + PARSES] edit confined between anchors.
  frozen prefix: 142 chars (unchanged)
  frozen suffix: 98 chars (unchanged)
  region: 31 chars -> 39 chars
```

Drop `--dry-run` to write the file in place. If the patched file no longer
parses, the CLI **refuses to write** (exit 3) unless you pass `--allow-broken`.

### Agent-callable

An agent shells out and consumes a stable JSON contract — no screen-scraping:

```bash
vibe-fim patch --stdin --before "…" --after "…" --name app.py --json
# {"bounded": true, "parses": true, "scope_report": "...", "unified_diff": "...",
#  "frozen_prefix_chars": 142, "frozen_suffix_chars": 98, "trimmed_overlap": 0,
#  "new_middle": "...", "wrote": false}
```

`--stdin`/`--text` pass the source inline (no disk round-trip); on error the
record is `{"bounded": false, "error": "..."}` with a non-zero exit, so a loop
can branch. There is also an MCP server — see **Wiring into an agent** below.

## Library

```python
from vibe_fim import patch_region

result = patch_region(
    source_code,
    before="def to_celsius(f):",   # frozen prefix ends right after this
    after="\n\ndef to_fahrenheit", # frozen suffix starts right at this
    hint="round the result to 1 decimal place",
)

assert result.bounded                         # structural invariant
assert source_code.startswith(result.prefix)  # prefix is byte-identical
assert result.parses is not False             # patched file still parses
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

## Verification that can fail

Bounded scope is structural (the prefix/suffix are the original slices). The
value is in the checks that a bad completion can actually trip:

- **Boundary echo.** FIM models sometimes re-emit the start of the suffix (or end
  of the prefix) inside the middle — duplicating code, on the same line, not just
  across a line break. `patch_region` detects and strips a real re-emission (a run
  of ≥4 non-whitespace chars) while leaving a coincidental shared `}` or a
  boundary newline alone. `result.trimmed_overlap` reports what was removed.
- **Truncation.** A completion cut off at `max_tokens` is an incomplete middle.
  `finish_reason == "length"` raises instead of shipping a half-written edit.
- **Parse gate.** For languages we can parse with the stdlib (Python via `ast`),
  the patched file is re-parsed; `result.parses` is `True`/`False`/`None`, the
  report shows `[BOUNDED + PARSES]` / `[BOUNDED + DOES NOT PARSE]`, and the CLI
  refuses to write a non-parsing result by default.

## The guarantee, and the drift — demonstrated

Two separate claims, kept separate:

- **The guarantee** is structural: surgical FIM changes **0** lines outside the
  target, by construction, because the frozen regions are never regenerated.
- **The drift** is measured. `examples/bench.py` replays a **committed Codestral
  transcript** (`examples/fixtures/`) and counts collateral with one
  newline-aware counter for both approaches:

```
surgical FIM      : 0 lines changed outside target
block regeneration: 2 lines changed outside target   # one source line: its trailing newline dropped (counted as a -/+ diff pair)
```

It runs **offline** by default (no key, same numbers for every reviewer); pass
`--live` to re-capture from the API. The block-regen baseline produced the
*correct* body and still silently altered the file outside the edit.

```bash
python examples/bench.py            # offline replay of the committed fixture
MISTRAL_API_KEY=... python examples/bench.py --live
```

## Wiring into an agent

`vibe-fim patch --json` is the stable contract for shelling out. For a native
tool, install the optional MCP server and point any MCP client at it:

```bash
pip install "vibe-fim[mcp]"
vibe-fim-mcp          # stdio MCP server exposing one tool: surgical_patch
```

The tool returns the unified diff plus the scope report, so the agent sees the
bounded proof — not just a success flag. (The base package stays
zero-dependency; the `mcp` SDK is pulled in only by the `[mcp]` extra.)

## How it works

1. Locate the unique `before` and `after` anchors.
2. `prefix = text[:end_of_before]`, `suffix = text[start_of_after:]`.
3. Send `(prefix, suffix)` to Codestral FIM → `middle` (de-duplicated, truncation-checked).
4. `new_text = prefix + middle + suffix`; the bytes outside the region are the
   same slices, concatenated unchanged.
5. Run the parse gate and surface the result.

## License

MIT © Guillaume Vele
