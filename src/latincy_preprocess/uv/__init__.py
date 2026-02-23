"""
U/V normalization submodule.

Re-exports the rule engine and applies Rust backend acceleration
when available.
"""

from latincy_preprocess.uv._rules import (
    Change,
    NormalizationResult,
    UVNormalizerRules,
    normalize_uv,
    normalize_vu,
    _get_context,
)

__all__ = [
    "UVNormalizerRules",
    "NormalizationResult",
    "Change",
    "normalize_uv",
    "normalize_vu",
]


def _apply_rust_backend(rust_module):
    """Monkey-patch Rust implementations onto UVNormalizerRules."""

    def _rust_normalize(self, text: str) -> str:
        if not text:
            return text
        return rust_module.normalize_uv(text)

    def _rust_normalize_char(self, text: str, idx: int) -> tuple[str, str]:
        char = text[idx]
        if char.lower() not in ("u", "v"):
            raise ValueError(f"Character at position {idx} is not u/v: '{char}'")
        return rust_module.normalize_uv_char(text, idx)

    def _rust_normalize_detailed(self, text: str) -> NormalizationResult:
        if not text:
            return NormalizationResult(original=text, normalized=text, changes=[])
        result = rust_module.normalize_uv_detailed(text)
        changes = [
            Change(
                position=c["position"],
                original=c["original"],
                normalized=c["normalized"],
                rule=c["rule"],
                context=c["context"],
            )
            for c in result["changes"]
        ]
        return NormalizationResult(
            original=result["original"],
            normalized=result["normalized"],
            changes=changes,
        )

    UVNormalizerRules.normalize = _rust_normalize
    UVNormalizerRules.normalize_char = _rust_normalize_char
    UVNormalizerRules.normalize_detailed = _rust_normalize_detailed
