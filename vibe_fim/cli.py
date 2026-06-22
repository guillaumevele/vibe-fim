"""Command-line interface for vibe-fim."""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from . import __version__
from .fim import DEFAULT_MODEL, FimError, patch_region


def _diff(old: str, new: str, path: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vibe-fim",
        description="Bounded AI code edits via Codestral fill-in-the-middle.",
    )
    parser.add_argument("--version", action="version", version=f"vibe-fim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("patch", help="rewrite the region between two anchors, scope-bounded")
    p.add_argument("--file", required=True, help="file to edit")
    p.add_argument("--before", required=True, help="unique anchor; frozen prefix ends right after it")
    p.add_argument("--after", required=True, help="unique anchor; frozen suffix starts right at it")
    p.add_argument("--hint", default="", help="optional steer (sent as a comment, not written)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Codestral FIM model")
    p.add_argument("--dry-run", action="store_true", help="show the diff, do not write")

    args = parser.parse_args(argv)
    path = Path(args.file)
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"vibe-fim: cannot read {args.file}: {exc}", file=sys.stderr)
        return 2

    try:
        result = patch_region(
            original, args.before, args.after,
            hint=args.hint,
            fim_fn=lambda pre, suf: __import__("vibe_fim.fim", fromlist=["codestral_fim"])
            .codestral_fim(pre, suf, model=args.model),
        )
    except FimError as exc:
        print(f"vibe-fim: {exc}", file=sys.stderr)
        return 1

    print(result.scope_report(), file=sys.stderr)
    diff = _diff(original, result.text, args.file)
    if args.dry_run:
        sys.stdout.write(diff or "(no change)\n")
        return 0
    path.write_text(result.text, encoding="utf-8")
    sys.stderr.write(f"wrote {args.file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
