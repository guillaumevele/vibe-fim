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

    pm = sub.add_parser("patch-many",
                        help="apply N bounded edits as one all-or-nothing transaction")
    msrc = pm.add_mutually_exclusive_group(required=True)
    msrc.add_argument("--file", help="file to edit in place")
    msrc.add_argument("--text", help="edit this literal text instead of a file")
    msrc.add_argument("--stdin", action="store_true", help="read the text from stdin")
    esrc = pm.add_mutually_exclusive_group(required=True)
    esrc.add_argument("--edits", help='JSON: [{"before":..,"after":..,"hint":..}, ...] in file order')
    esrc.add_argument("--edits-file", help="path to that JSON")
    pm.add_argument("--model", default=DEFAULT_MODEL)
    pm.add_argument("--max-tokens", type=int, default=1024)
    pm.add_argument("--temperature", type=float, default=0.0)
    pm.add_argument("--name", help="filename hint for the parser/comment style")
    pm.add_argument("--json", action="store_true")
    pm.add_argument("--dry-run", action="store_true")
    pm.add_argument("--allow-broken", action="store_true")

    g = sub.add_parser("grow-tests",
                       help="append PROVABLY-additive tests (old tests stay green by execution)")
    gsrc = g.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--file", help="pytest file to grow in place")
    gsrc.add_argument("--text", help="pytest source instead of a file")
    g.add_argument("--focus", default="", help="what to test (sent as a steer)")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--name", help="filename hint (with --text)")
    g.add_argument("--no-run", action="store_true", help="skip the pytest proof (then it is only INSERTED, not proven additive)")
    g.add_argument("--json", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--allow-broken", action="store_true", help="write even if not provably additive")

    args = parser.parse_args(argv)
    if args.command == "patch-many":
        return _cmd_patch_many(args)
    if args.command == "grow-tests":
        return _cmd_grow_tests(args)

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

    # The headline verification failed — never emit the broken result anywhere a
    # caller might capture it (a written file OR stdout, which an agent may pipe
    # to a file), on ANY input path. Signal with exit 3.
    if result.parses is False and not args.allow_broken:
        msg = "patched result does not parse (use --allow-broken to override)"
        if args.json:
            return _fail(msg, True, code=3, result=result, diff=diff)
        print(result.scope_report(), file=sys.stderr)
        print(diff, file=sys.stderr)
        print(f"vibe-fim: {msg}", file=sys.stderr)
        return 3

    will_write = bool(args.file) and not args.dry_run

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


def _resolve_source(args):
    """Return (original_text, name, path_or_None) for --file/--text/--stdin."""
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8"), args.file, Path(args.file)
        except OSError as exc:
            raise FimError(f"cannot read {args.file}: {exc}")
    if getattr(args, "stdin", False):
        return sys.stdin.read(), args.name, None
    return args.text, args.name, None


def _cmd_patch_many(args) -> int:
    from .fim import patch_regions

    try:
        original, name, path = _resolve_source(args)
    except FimError as exc:
        return _fail(str(exc), args.json, code=2)

    raw = args.edits if args.edits else Path(args.edits_file).read_text(encoding="utf-8")
    try:
        edits = json.loads(raw)
        assert isinstance(edits, list)
    except (ValueError, AssertionError):
        return _fail("--edits must be a JSON array of {before, after, hint}", args.json, code=2)

    backend = functools.partial(codestral_fim, model=args.model,
                                max_tokens=args.max_tokens, temperature=args.temperature)
    try:
        result = patch_regions(original, edits, fim_fn=backend, filename=name)
    except FimError as exc:
        return _fail(str(exc), args.json, code=1)

    diff = _diff(original, result.text, name or "input")
    if result.parses is False and not args.allow_broken:
        msg = "transaction rejected: patched result does not parse (use --allow-broken)"
        if args.json:
            json.dump({"bounded": result.bounded, "parses": False, "regions": result.region_count,
                       "error": msg, "unified_diff": diff}, sys.stdout)
            sys.stdout.write("\n")
            return 3
        print(result.scope_report(), file=sys.stderr)
        print(diff, file=sys.stderr)
        print(f"vibe-fim: {msg}", file=sys.stderr)
        return 3

    will_write = bool(args.file) and not args.dry_run
    if will_write:
        path.write_text(result.text, encoding="utf-8")

    if args.json:
        json.dump({"bounded": result.bounded, "parses": result.parses,
                   "regions": result.region_count, "trimmed_overlap": result.trimmed_overlap,
                   "scope_report": result.scope_report(), "unified_diff": diff,
                   "wrote": will_write}, sys.stdout)
        sys.stdout.write("\n")
        return 0

    print(result.scope_report(), file=sys.stderr)
    if will_write:
        sys.stderr.write(f"wrote {args.file} ({result.region_count} regions)\n")
    elif args.file and args.dry_run:
        sys.stdout.write(diff or "(no change)\n")
    else:
        sys.stdout.write(result.text)
    return 0


def _cmd_grow_tests(args) -> int:
    from .testgen import grow_tests, pytest_runner

    try:
        original, name, path = _resolve_source(args)
    except FimError as exc:
        return _fail(str(exc), args.json, code=2)
    name = name or "tests.py"

    backend = functools.partial(codestral_fim, model=args.model,
                                max_tokens=args.max_tokens, temperature=args.temperature)
    runner = None if args.no_run else pytest_runner
    try:
        result = grow_tests(original, filename=name, focus=args.focus,
                            fim_fn=backend, runner=runner)
    except FimError as exc:
        return _fail(str(exc), args.json, code=1)

    diff = _diff(original, result.text, name)
    # Exit code: 3 if it does not parse, 4 if the additive proof failed, else 0.
    blocked = (not result.additive) and not args.allow_broken
    code = 0
    if blocked:
        code = 3 if result.parses is False else 4

    if args.json:
        json.dump({"additive": result.additive, "prefix_identical": result.prefix_identical,
                   "parses": result.parses, "ids_monotonic": result.ids_monotonic,
                   "green_preserved": result.green_preserved, "ran": result.ran,
                   "added": list(result.added_ids), "removed": list(result.removed_ids),
                   "report": result.report(), "unified_diff": diff,
                   "wrote": bool(args.file) and not args.dry_run and not blocked}, sys.stdout)
        sys.stdout.write("\n")
        return code

    print(result.report(), file=sys.stderr)
    if blocked:
        print(diff, file=sys.stderr)
        print(f"vibe-fim: not provably additive — not writing (use --allow-broken)", file=sys.stderr)
        return code
    if args.file and not args.dry_run:
        path.write_text(result.text, encoding="utf-8")
        sys.stderr.write(f"wrote {args.file} (+{len(result.added_ids)} tests)\n")
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
