"""vibe-fim — bounded AI code edits via Codestral fill-in-the-middle.

Edit only the region between two anchors; the rest of the file is provably
byte-identical. A surgical alternative to block regeneration, powered by
Mistral Codestral FIM.
"""
from __future__ import annotations

from .fim import (
    DEFAULT_MODEL,
    FimError,
    MultiPatchResult,
    PatchResult,
    codestral_fim,
    patch_region,
    patch_regions,
)

__version__ = "0.3.0"

__all__ = [
    "patch_region",
    "patch_regions",
    "codestral_fim",
    "PatchResult",
    "MultiPatchResult",
    "FimError",
    "DEFAULT_MODEL",
    "__version__",
]
