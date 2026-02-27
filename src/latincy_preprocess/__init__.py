"""
latincy-preprocess: Latin text preprocessing.

Consolidates U/V normalization and long-s OCR correction into a single
package with optional Rust acceleration.

Basic usage:
    >>> from latincy_preprocess import normalize
    >>> normalize("ftatua uirumque cano")
    'statua virumque cano'

Per-normalizer usage:
    >>> from latincy_preprocess.uv import normalize_uv
    >>> normalize_uv("Arma uirumque cano")
    'Arma virumque cano'

    >>> from latincy_preprocess.long_s import LongSNormalizer
    >>> n = LongSNormalizer()
    >>> n.normalize_word_full("ftatua")
    ('statua', [...])
"""

from latincy_preprocess.uv import (
    Change,
    NormalizationResult,
    UVNormalizerRules,
    normalize_uv,
    normalize_vu,
)
from latincy_preprocess.long_s import LongSNormalizer, TransformationRule
from latincy_preprocess._strip import strip_macrons
from latincy_preprocess.diacritics import strip_diacritics

try:
    from latincy_preprocess import _rust
    _BACKEND = "rust"
except ImportError:
    _rust = None
    _BACKEND = "python"

__version__ = "0.1.2"
__all__ = [
    "normalize",
    "backend",
    "UVNormalizerRules",
    "NormalizationResult",
    "Change",
    "normalize_uv",
    "normalize_vu",
    "LongSNormalizer",
    "TransformationRule",
    "strip_macrons",
    "strip_diacritics",
]


def backend() -> str:
    """Return 'rust' or 'python' indicating the active backend."""
    return _BACKEND


def normalize(text: str) -> str:
    """
    Apply all normalizations to Latin text.

    Order: long-s first (OCR correction), then U/V (phonological).
    This order matters: ſtatua → statua → statua (no u/v change),
    but uirumque → uirumque (no long-s) → virumque.

    Args:
        text: Latin text (may contain OCR artifacts and u-only spelling)

    Returns:
        Normalized text with long-s corrected and proper u/v distinction
    """
    # Pass 1: Long-s correction
    normalizer = LongSNormalizer()
    text = normalizer.normalize_text_full(text, apply_pass2=True)
    # Pass 2: U/V normalization
    text = normalize_uv(text)
    return text


# Apply Rust backend if available
if _rust is not None:
    from latincy_preprocess.uv import _apply_rust_backend as _apply_uv_rust
    from latincy_preprocess.long_s import _apply_rust_backend as _apply_long_s_rust
    _apply_uv_rust(_rust)
    _apply_long_s_rust(_rust)


# Lazy import for spaCy components (only when spacy is installed)
def __getattr__(name: str):
    if name == "LatinPreprocessorComponent":
        try:
            from latincy_preprocess.spacy import LatinPreprocessorComponent
            return LatinPreprocessorComponent
        except ImportError:
            raise ImportError(
                "spaCy integration requires spacy. "
                "Install with: pip install latincy-preprocess[spacy]"
            )
    if name == "UVNormalizerComponent":
        try:
            from latincy_preprocess.spacy import UVNormalizerComponent
            return UVNormalizerComponent
        except ImportError:
            raise ImportError(
                "spaCy integration requires spacy. "
                "Install with: pip install latincy-preprocess[spacy]"
            )
    if name == "LongSNormalizerComponent":
        try:
            from latincy_preprocess.spacy import LongSNormalizerComponent
            return LongSNormalizerComponent
        except ImportError:
            raise ImportError(
                "spaCy integration requires spacy. "
                "Install with: pip install latincy-preprocess[spacy]"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
