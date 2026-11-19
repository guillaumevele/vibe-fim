"""The MCP tools must stay inside the workspace root."""
import os
import sys
import types

import pytest

from vibe_fim import mcp_server
from vibe_fim.workspace import WorkspaceError, resolve_in_workspace


def test_relative_and_absolute_paths_inside_root(tmp_path):
    (tmp_path / "pkg").mkdir()
    target = tmp_path / "pkg" / "a.py"
    target.write_text("x = 1\n")
    assert resolve_in_workspace("pkg/a.py", tmp_path) == target.resolve()
    assert resolve_in_workspace(str(target), tmp_path) == target.resolve()


@pytest.mark.parametrize("bad", ["../outside.py", "pkg/../../outside.py", "", "a\x00b"])
def test_paths_escaping_the_root_are_refused(tmp_path, bad):
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (tmp_path / "outside.py").write_text("secret = 1\n")
    with pytest.raises(WorkspaceError):
        resolve_in_workspace(bad, root)


def test_absolute_path_outside_root_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    with pytest.raises(WorkspaceError):
        resolve_in_workspace(str(outside), root)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_symlink_pointing_outside_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    (root / "link.py").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        resolve_in_workspace("link.py", root)


def test_sibling_directory_with_same_prefix_is_refused(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    sibling = tmp_path / "proj-secrets"
    sibling.mkdir()
    (sibling / "k.py").write_text("k = 1\n")
    with pytest.raises(WorkspaceError):
        resolve_in_workspace(str(sibling / "k.py"), root)


class _FakeMCP:
    """Stands in for mcp.server.fastmcp.FastMCP so the tools run without the SDK."""

    def __init__(self, name):
        self.tools = {}

    def tool(self):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture
def tools(monkeypatch, tmp_path):
    fake = types.ModuleType("mcp.server.fastmcp")
    fake.FastMCP = _FakeMCP
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake)
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("VIBE_FIM_ROOT", str(root))
    calls = []

    def backend(prefix, suffix, **kwargs):
        calls.append((prefix, suffix))
        return "\n    return 2\n"

    monkeypatch.setattr(mcp_server, "codestral_fim", backend)
    return mcp_server._build_server().tools, root, calls


def test_tools_never_read_or_send_a_file_outside_the_root(tools, tmp_path):
    registered, root, calls = tools
    outside = tmp_path / "outside.py"
    outside.write_text("def f():\n    return 1\n# end\n")
    before = outside.read_bytes()
    edits = [{"before_anchor": "def f():", "after_anchor": "# end"}]
    results = [
        registered["surgical_patch"](str(outside), "def f():", "# end", dry_run=False),
        registered["surgical_patch"]("../outside.py", "def f():", "# end", dry_run=False),
        registered["surgical_patch_many"](str(outside), edits, dry_run=False),
        registered["grow_tests"](str(outside), dry_run=False),
    ]
    for result in results:
        assert result.startswith("ERROR:") and "outside the workspace root" in result
    assert calls == []  # nothing was sent to the backend
    assert outside.read_bytes() == before  # nothing was written


def test_tool_still_edits_a_file_inside_the_root(tools):
    registered, root, calls = tools
    target = root / "m.py"
    target.write_text("def f():\n    return 1\n# end\n")
    result = registered["surgical_patch"]("m.py", "def f():", "# end", dry_run=False)
    assert "wrote file" in result
    assert len(calls) == 1
    assert target.read_text() == "def f():\n    return 2\n# end\n"
