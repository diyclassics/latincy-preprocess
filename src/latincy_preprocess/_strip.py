"""
Stripping utilities for Latin text.

Provides lightweight functions that remove diacritical marks from Latin
text, useful for preparing input to normalizer models or for search
indexing.

These have no external dependencies — pure Python + unicodedata.
"""

from __future__ import annotations

import unicodedata

__all__ = ["strip_macrons"]

# Unicode combining marks used for Latin macrons/breves
_MACRON_MARKS = {
    "\u0304",  # combining macron
    "\u0306",  # combining breve (sometimes paired with macrons in pedagogical texts)
}

# Precomposed macron vowels → base vowels (NFC forms)
_MACRON_MAP = str.maketrans(
    "āēīōūȳĀĒĪŌŪȲ",
    "aeiouyAEIOUY",
)


def strip_macrons(text: str) -> str:
    """
    Remove macrons (and breves) from Latin text, preserving case.

    Handles both precomposed characters (ā → a) and combining marks
    (a + U+0304 → a).

    Args:
        text: Latin text (possibly with macrons)

    Returns:
        Text with macrons removed

    Example:
        >>> strip_macrons("laudāre")
        'laudare'
        >>> strip_macrons("Rōma")
        'Roma'
    """
    # Fast path: translate precomposed macron characters
    text = text.translate(_MACRON_MAP)
    # Slow path: decompose and strip combining macron marks
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if c not in _MACRON_MARKS)
    return unicodedata.normalize("NFC", stripped)
