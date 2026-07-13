"""UTF-8 source I/O that preserves the file's existing line endings."""
from __future__ import annotations

from pathlib import Path


def read_utf8_preserving_newlines(path: Path) -> str:
    """Read UTF-8 text without translating CRLF or CR line endings."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def write_utf8_preserving_newlines(path: Path, text: str) -> None:
    """Write UTF-8 text without platform newline translation."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
