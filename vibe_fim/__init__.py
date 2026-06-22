"""vibe-fim — bounded AI code edits via Codestral fill-in-the-middle.

Edit only the region between two anchors; the rest of the file is provably
byte-identical. A surgical alternative to block regeneration, powered by
Mistral Codestral FIM.
"""
from __future__ import annotations

from .fim import (
    DEFAULT_MODEL,
    FimError,
    PatchResult,
    codestral_fim,
    patch_region,
)

__version__ = "0.2.0"

__all__ = [
    "patch_region",
    "codestral_fim",
    "PatchResult",
    "FimError",
    "DEFAULT_MODEL",
    "__version__",
]
