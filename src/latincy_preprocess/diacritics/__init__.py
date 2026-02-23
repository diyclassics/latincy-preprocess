"""
Diacritics utilities submodule.

Provides character-level diacritic stripping and analysis for Ancient Greek.
Restoration models live in a separate package (latincy-diacritics).

Basic usage:
    >>> from latincy_preprocess.diacritics import strip_diacritics
    >>> strip_diacritics("ἄνθρωπος")
    'ανθρωπος'

    >>> from latincy_preprocess.diacritics import base_char
    >>> base_char("ἄ")
    'α'
"""

from latincy_preprocess.diacritics._charset import (
    CharsetMap,
    GREEK_BASE_CHARS,
    MUTABLE_CHARS,
    aligned_pairs,
    base_char,
    build_charset,
    strip_diacritics,
)

__all__ = [
    "CharsetMap",
    "GREEK_BASE_CHARS",
    "MUTABLE_CHARS",
    "aligned_pairs",
    "base_char",
    "build_charset",
    "strip_diacritics",
]
