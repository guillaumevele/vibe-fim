"""Command-line interface for vibe-fim."""
from __future__ import annotations

import argparse
import difflib
import functools
import json
import sys
from pathlib import Path

from . import __version__
from .fim import DEFAULT_MODEL, FimError, codestral_fim, patch_region


def _diff(old: str, new: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def _json_record(result, *, diff: str, wrote: bool) -> dict:
    """The stable machine contract an agent consumes."""
    return {
        "bounded": result.bounded,
        "parses": result.parses,
        "scope_report": result.scope_report(),
        "unified_diff": diff,
        "new_middle": result.new_middle,
        "old_middle": result.old_middle,
        "frozen_prefix_chars": len(result.prefix),
        "frozen_suffix_chars": len(result.suffix),
        "trimmed_overlap": result.trimmed_overlap,
        "wrote": wrote,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vibe-fim",
        description="Bounded AI code edits via Codestral fill-in-the-middle.",
    )
    parser.add_argument("--version", action="version", version=f"vibe-fim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("patch", help="rewrite the region between two anchors, scope-bounded")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="file to edit in place")
    src.add_argument("--text", help="edit this literal text instead of a file (prints result)")
    src.add_argument("--stdin", action="store_true", help="read the text to edit from stdin")
    p.add_argument("--before", required=True, help="unique anchor; frozen prefix ends right after it")
    p.add_argument("--after", required=True, help="unique anchor; frozen suffix starts right at it")
    p.add_argument("--hint", default="", help="optional steer (sent as a comment, not written)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Codestral FIM model")
    p.add_argument("--max-tokens", type=int, default=1024, help="max tokens for the middle")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    p.add_argument("--name", help="filename hint for the parser/comment style (with --text/--stdin)")
    p.add_argument("--json", action="store_true", help="emit the machine contract as JSON")
    p.add_argument("--dry-run", action="store_true", help="do not write; show the diff")
    p.add_argument("--allow-broken", action="store_true", help="write even if the result no longer parses")

    args = parser.parse_args(argv)

    # Resolve the source text and the filename used for parsing / comment style.
    if args.file:
        path = Path(args.file)
        try:
            original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return _fail(f"cannot read {args.file}: {exc}", args.json, code=2)
        name = args.file
    elif args.stdin:
        original = sys.stdin.read()
        name = args.name
    else:
        original = args.text
        name = args.name

    backend = functools.partial(codestral_fim, model=args.model,
                                max_tokens=args.max_tokens, temperature=args.temperature)
    try:
        result = patch_region(original, args.before, args.after,
                              hint=args.hint, fim_fn=backend, filename=name)
    except FimError as exc:
        return _fail(str(exc), args.json, code=1)

    diff = _diff(original, result.text, name or "input")
    parse_failed = result.parses is False
    will_write = bool(args.file) and not args.dry_run and (not parse_failed or args.allow_broken)

    if args.file and parse_failed and not args.allow_broken and not args.dry_run:
        # The headline verification just failed — refuse to write by default.
        print(result.scope_report(), file=sys.stderr)
        return _fail("patched result does not parse; not writing (use --allow-broken)",
                     args.json, code=3, result=result, diff=diff)

    if will_write:
        path.write_text(result.text, encoding="utf-8")

    if args.json:
        json.dump(_json_record(result, diff=diff, wrote=will_write), sys.stdout)
        sys.stdout.write("\n")
        return 0

    print(result.scope_report(), file=sys.stderr)
    if will_write:
        sys.stderr.write(f"wrote {args.file}\n")
    elif args.file and args.dry_run:
        sys.stdout.write(diff or "(no change)\n")
    else:
        sys.stdout.write(result.text)
    return 0


def _fail(message: str, as_json: bool, *, code: int, result=None, diff: str = "") -> int:
    if as_json:
        record = {"bounded": bool(result and result.bounded), "error": message}
        if result is not None:
            record.update(_json_record(result, diff=diff, wrote=False))
            record["error"] = message
        json.dump(record, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"vibe-fim: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
