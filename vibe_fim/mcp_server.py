"""A tiny MCP stdio server exposing vibe-fim's bounded edit as one tool.

Install with the optional extra and run it; point any MCP client (including
Mistral `vibe`) at it:

    pip install "vibe-fim[mcp]"
    vibe-fim-mcp

The base package stays dependency-free — `mcp` is imported here only, and only
pulled in by the `[mcp]` extra.
"""
from __future__ import annotations

import difflib
import functools
import sys

from .fim import FimError, codestral_fim, patch_region
from .text_io import read_utf8_preserving_newlines, write_utf8_preserving_newlines
from .workspace import WorkspaceError, resolve_in_workspace


def _build_server():
    if sys.version_info < (3, 10):  # pragma: no cover - version-specific guard
        raise SystemExit(
            "vibe-fim-mcp requires Python 3.10 or newer; "
            "the core vibe-fim CLI continues to support Python 3.9."
        )
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise SystemExit(
            "vibe-fim-mcp needs the MCP SDK. Install it with:\n"
            '    pip install "vibe-fim[mcp]"'
        ) from exc

    server = FastMCP("vibe-fim")

    @server.tool()
    def surgical_patch(
        file: str,
        before_anchor: str,
        after_anchor: str,
        instruction: str = "",
        model: str = "codestral-latest",
        dry_run: bool = True,
    ) -> str:
        """Rewrite ONLY the region between two unique anchors using Codestral
        fill-in-the-middle. The prefix and suffix are byte-identical to the
        original — the edit is provably bounded. Returns the scope report and
        unified diff so the edit's bounds are visible, not just a success flag.

        Parse safety net: for Python (.py) the patched file is re-parsed and a
        non-parsing result is NOT written. For other languages the result is
        written without a syntactic check (the report shows no PARSES tag) —
        review the returned diff.

        - file:          path to the file to edit, inside the workspace root
                         (the server's start directory, or $VIBE_FIM_ROOT).
        - before_anchor: unique line; the frozen prefix ends right after it.
        - after_anchor:  unique line; the frozen suffix starts right at it.
        - instruction:   optional steer (sent as a comment, never written).
        - dry_run:       when true (default), do not write — only show the diff.
        """
        try:
            path = resolve_in_workspace(file)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        try:
            original = read_utf8_preserving_newlines(path)
        except (OSError, UnicodeDecodeError) as exc:
            return f"ERROR: cannot read {file}: {exc}"

        backend = functools.partial(codestral_fim, model=model)
        try:
            result = patch_region(original, before_anchor, after_anchor,
                                  hint=instruction, fim_fn=backend, filename=file)
        except FimError as exc:
            return f"ERROR: {exc}"

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            result.text.splitlines(keepends=True),
            fromfile=f"a/{file}", tofile=f"b/{file}",
        ))
        wrote = False
        if not dry_run and result.parses is not False:
            write_utf8_preserving_newlines(path, result.text)
            wrote = True

        status = "wrote file" if wrote else "dry run (not written)"
        if result.parses is False:
            status = "NOT written — patched result does not parse"
        return f"{result.scope_report()}\n\nstatus: {status}\n\n{diff or '(no change)'}"

    @server.tool()
    def surgical_patch_many(
        file: str,
        edits: list,
        model: str = "codestral-latest",
        dry_run: bool = True,
    ) -> str:
        """Apply N bounded FIM edits to one file as a single ALL-OR-NOTHING
        transaction. `edits` is a list of {before_anchor, after_anchor,
        instruction?} in file order, non-overlapping. Every segment between and
        around the regions stays byte-identical, the assembled file is
        parse-checked once, and the whole result is written or nothing is — a
        model drifting in one region cannot corrupt the others. Returns the scope
        report + unified diff."""
        from .fim import patch_regions

        try:
            path = resolve_in_workspace(file)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        try:
            original = read_utf8_preserving_newlines(path)
        except (OSError, UnicodeDecodeError) as exc:
            return f"ERROR: cannot read {file}: {exc}"
        spec = [{"before": e.get("before_anchor", e.get("before", "")),
                 "after": e.get("after_anchor", e.get("after", "")),
                 "hint": e.get("instruction", e.get("hint", ""))} for e in edits]
        backend = functools.partial(codestral_fim, model=model)
        try:
            result = patch_regions(original, spec, fim_fn=backend, filename=file)
        except FimError as exc:
            return f"ERROR: {exc}"

        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), result.text.splitlines(keepends=True),
            fromfile=f"a/{file}", tofile=f"b/{file}"))
        wrote = False
        if not dry_run and result.parses is not False:
            write_utf8_preserving_newlines(path, result.text)
            wrote = True
        status = "wrote file" if wrote else "dry run (not written)"
        if result.parses is False:
            status = "NOT written — transaction rejected (result does not parse)"
        return f"{result.scope_report()}\n\nstatus: {status}\n\n{diff or '(no change)'}"

    @server.tool()
    def grow_tests(file: str, focus: str = "", model: str = "codestral-latest",
                   dry_run: bool = True) -> str:
        """Append PROVABLY-additive pytest tests to a file. Generates new test_*
        functions via Codestral FIM and proves the change is additive: the
        pre-existing tests are byte-identical, the file parses, and a real pytest
        run shows every old test kept its outcome while the new tests executed. If
        the proof fails, nothing is written. Generated tests are CANDIDATES —
        review them for meaning. Returns the proof report + node-id delta + diff."""
        from .testgen import grow_tests as _grow, pytest_runner

        try:
            path = resolve_in_workspace(file)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        try:
            original = read_utf8_preserving_newlines(path)
        except (OSError, UnicodeDecodeError) as exc:
            return f"ERROR: cannot read {file}: {exc}"
        backend = functools.partial(codestral_fim, model=model)
        try:
            result = _grow(original, filename=file, focus=focus,
                           fim_fn=backend, runner=pytest_runner)
        except FimError as exc:
            return f"ERROR: {exc}"
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), result.text.splitlines(keepends=True),
            fromfile=f"a/{file}", tofile=f"b/{file}"))
        wrote = False
        if not dry_run and result.additive:
            write_utf8_preserving_newlines(path, result.text)
            wrote = True
        status = "wrote file" if wrote else (
            "NOT written — not provably additive" if not result.additive else "dry run (not written)")
        return f"{result.report()}\n\nstatus: {status}\n\n{diff or '(no change)'}"

    return server


def main() -> None:
    _build_server().run()


if __name__ == "__main__":
    main()
