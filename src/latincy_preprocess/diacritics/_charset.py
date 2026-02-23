"""
Character set definitions and diacritic stripping for Ancient Greek.

Provides:
- strip_diacritics(): remove all Greek diacritics, lowercase
- CharsetMap: bidirectional char↔index mapping with save/load
- build_charset(): scan corpus to build complete mappings
- Per-character diacritic class definitions
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Sequence

__all__ = [
    "strip_diacritics",
    "CharsetMap",
    "build_charset",
    "GREEK_BASE_CHARS",
    "base_char",
]

# The 25 base Greek lowercase letters (24 + final sigma)
GREEK_BASE_CHARS = list("αβγδεζηθικλμνξοπρσςτυφχψω")

# Characters that can carry diacritics (vowels + rho)
MUTABLE_CHARS = set("αεηιουωρ")

# Unicode combining marks used in Greek polytonic
_COMBINING_MARKS = {
    "\u0300",  # combining grave accent
    "\u0301",  # combining acute accent
    "\u0302",  # combining circumflex
    "\u0303",  # combining tilde
    "\u0304",  # combining macron
    "\u0306",  # combining breve
    "\u0308",  # combining diaeresis
    "\u0313",  # combining comma above (smooth breathing)
    "\u0314",  # combining reversed comma above (rough breathing)
    "\u0342",  # combining Greek perispomeni (circumflex)
    "\u0345",  # combining Greek ypogegrammeni (iota subscript)
}


def strip_diacritics(text: str) -> str:
    """
    Remove all Greek diacritics and lowercase the text.

    Decomposes polytonic characters to base + combining marks, removes
    the combining marks, then recomposes. Non-Greek characters pass through.

    Args:
        text: Greek text (possibly with polytonic diacritics)

    Returns:
        Stripped, lowercased text with only base Greek letters
    """
    text = text.lower()
    # Decompose to NFD (base char + combining marks)
    decomposed = unicodedata.normalize("NFD", text)
    # Remove all combining marks used in Greek
    stripped = "".join(c for c in decomposed if c not in _COMBINING_MARKS)
    # Recompose (handles any remaining composed chars)
    return unicodedata.normalize("NFC", stripped)


def base_char(ch: str) -> str:
    """
    Return the base character for a (possibly accented) Greek character.

    Args:
        ch: A single character

    Returns:
        The base character with all diacritics removed
    """
    decomposed = unicodedata.normalize("NFD", ch.lower())
    base = "".join(c for c in decomposed if c not in _COMBINING_MARKS)
    return unicodedata.normalize("NFC", base)


class CharsetMap:
    """
    Bidirectional mapping between characters and integer indices.

    Maintains:
    - input_to_idx / idx_to_input: stripped input alphabet
    - output_to_idx / idx_to_output: polytonic output alphabet
    - base_to_variants: for each base char, all observed polytonic forms
    - num_classes_per_char: for per-position classification

    The output vocabulary is organized so that each base character maps
    to a contiguous block of diacritic variants, enabling per-character
    classification (each input position predicts from its own class set).

    For simplicity, we use a single flat output vocabulary with a global
    softmax. The per-character variant mapping is available for analysis.
    """

    def __init__(
        self,
        input_chars: list[str],
        output_chars: list[str],
        base_to_variants: dict[str, list[str]],
    ) -> None:
        self.input_chars = input_chars
        self.output_chars = output_chars
        self.base_to_variants = base_to_variants

        # Build index maps
        self.input_to_idx = {c: i for i, c in enumerate(input_chars)}
        self.idx_to_input = {i: c for i, c in enumerate(input_chars)}
        self.output_to_idx = {c: i for i, c in enumerate(output_chars)}
        self.idx_to_output = {i: c for i, c in enumerate(output_chars)}

    @property
    def input_size(self) -> int:
        return len(self.input_chars)

    @property
    def output_size(self) -> int:
        return len(self.output_chars)

    def encode_input(self, text: str) -> list[int]:
        """Convert stripped text to list of input indices."""
        unk = self.input_to_idx.get("<unk>", 0)
        return [self.input_to_idx.get(c, unk) for c in text]

    def encode_output(self, text: str) -> list[int]:
        """Convert polytonic text to list of output indices."""
        unk = self.output_to_idx.get("<unk>", 0)
        return [self.output_to_idx.get(c, unk) for c in text]

    def decode_output(self, indices: list[int]) -> str:
        """Convert list of output indices back to text."""
        return "".join(self.idx_to_output.get(i, "?") for i in indices)

    def save(self, path: str | Path) -> None:
        """Save charset to JSON file."""
        data = {
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "base_to_variants": self.base_to_variants,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CharsetMap:
        """Load charset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            input_chars=data["input_chars"],
            output_chars=data["output_chars"],
            base_to_variants=data["base_to_variants"],
        )

    def __repr__(self) -> str:
        return (
            f"CharsetMap(input={self.input_size}, output={self.output_size}, "
            f"bases={len(self.base_to_variants)})"
        )


def aligned_pairs(text: str) -> list[tuple[str, str]]:
    """
    Generate aligned (base, polytonic) character pairs from polytonic text.

    NFC-normalizes the text, then maps each character to its base form.
    Bare combining marks (which have no precomposed form in NFC) are
    dropped to maintain 1:1 alignment.

    Args:
        text: Polytonic Greek text (lowercased)

    Returns:
        List of (base_char, polytonic_char) tuples with 1:1 alignment
    """
    nfc = unicodedata.normalize("NFC", text.lower())
    pairs = []
    for ch in nfc:
        if unicodedata.combining(ch):
            # Skip bare combining marks (no precomposed form exists)
            continue
        b = base_char(ch)
        if b:  # base_char returns "" for pure combining marks
            pairs.append((b, ch))
    return pairs


def build_charset(texts: Sequence[str]) -> CharsetMap:
    """
    Scan a corpus of polytonic Greek texts to build a complete CharsetMap.

    Uses per-character base extraction (not string-level stripping) to
    maintain correct alignment. Bare combining marks are dropped.

    Args:
        texts: Iterable of polytonic Greek text strings

    Returns:
        A CharsetMap with complete input/output mappings
    """
    base_to_variants: dict[str, set[str]] = defaultdict(set)
    all_output_chars: set[str] = set()
    all_input_chars: set[str] = set()

    for text in texts:
        for base, poly in aligned_pairs(text):
            all_input_chars.add(base)
            all_output_chars.add(poly)
            base_to_variants[base].add(poly)

    # Build sorted vocabularies with special tokens
    input_chars = ["<pad>", "<unk>"] + sorted(all_input_chars)
    output_chars = ["<pad>", "<unk>"] + sorted(all_output_chars)

    # Convert sets to sorted lists
    base_variants_sorted = {
        k: sorted(v) for k, v in sorted(base_to_variants.items())
    }

    return CharsetMap(
        input_chars=input_chars,
        output_chars=output_chars,
        base_to_variants=base_variants_sorted,
    )
