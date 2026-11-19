"""Workspace confinement for the MCP server.

An MCP tool call comes from a model, and a model can be steered by text it has
read. Without a boundary, one call could send any readable file (keys, shell
profiles, another project) to the FIM backend, or rewrite it. The MCP tools
therefore only touch files under one root: `VIBE_FIM_ROOT` when set, otherwise
the directory the server was started in.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "VIBE_FIM_ROOT"


class WorkspaceError(ValueError):
    pass


def workspace_root() -> Path:
    return Path(os.environ.get(ROOT_ENV) or os.getcwd()).resolve()


def resolve_in_workspace(file: str, root: Path | None = None) -> Path:
    """Return the real path of `file` if it lies under the workspace root.

    Relative paths are taken from the root. Symlinks are resolved before the
    check, so a link inside the root that points outside is refused too."""
    root = (root or workspace_root()).resolve()
    if not file or "\x00" in file:
        raise WorkspaceError("file path is empty or invalid")
    candidate = Path(file).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    real = candidate.resolve()
    if real != root and root not in real.parents:
        raise WorkspaceError(
            f"{file} is outside the workspace root {root}; "
            f"start vibe-fim-mcp in the project directory or set {ROOT_ENV}"
        )
    return real
