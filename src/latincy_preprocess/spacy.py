"""
spaCy integration for latincy-preprocess.

Provides pipeline components for U/V normalization and long-s correction.

Example:
    >>> import spacy
    >>> nlp = spacy.blank("la")
    >>> nlp.add_pipe("uv_normalizer")
    >>> doc = nlp("Arma uirumque cano")
    >>> doc._.uv_normalized
    'Arma virumque cano'
"""

from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc, Token

from latincy_preprocess.uv._rules import UVNormalizerRules
from latincy_preprocess.long_s._rules import LongSNormalizer

__all__ = [
    "LatinPreprocessorComponent",
    "UVNormalizerComponent",
    "LongSNormalizerComponent",
    "create_latin_preprocessor",
    "create_uv_normalizer",
    "create_long_s_normalizer",
]


# =============================================================================
# Unified Latin Preprocessor Component
# =============================================================================


@Language.factory(
    "latin_preprocessor",
    default_config={"long_s": True, "uv": True, "long_s_pass2": True},
    assigns=[
        "doc._.preprocessed",
        "token._.preprocessed",
        "token._.preprocessed_lemma",
    ],
)
def create_latin_preprocessor(
    nlp: Language,
    name: str,
    long_s: bool = True,
    uv: bool = True,
    long_s_pass2: bool = True,
) -> "LatinPreprocessorComponent":
    """Create a unified Latin preprocessor pipeline component.

    Chains long-s correction then U/V normalization in the correct order.
    Either normalizer can be disabled via config.
    """
    return LatinPreprocessorComponent(
        nlp, name, long_s=long_s, uv=uv, long_s_pass2=long_s_pass2
    )


class LatinPreprocessorComponent:
    """Unified spaCy pipeline component for Latin text preprocessing.

    Applies long-s OCR correction followed by U/V normalization in a single
    pass. Order matters: long-s first avoids interference (e.g., ſtatua →
    statua before U/V rules run).

    Extensions:
        - Doc._.preprocessed: Full preprocessed text.
        - Token._.preprocessed: Preprocessed token text.
        - Token._.preprocessed_lemma: Preprocessed lemma (v-space if uv=True).
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        long_s: bool = True,
        uv: bool = True,
        long_s_pass2: bool = True,
    ) -> None:
        self.name = name
        self.long_s = long_s
        self.uv = uv
        self.long_s_pass2 = long_s_pass2

        self._long_s_normalizer = LongSNormalizer() if long_s else None
        self._uv_normalizer = UVNormalizerRules() if uv else None

        if not Doc.has_extension("preprocessed"):
            Doc.set_extension("preprocessed", default=None)
        if not Token.has_extension("preprocessed"):
            Token.set_extension("preprocessed", default=None)
        if not Token.has_extension("preprocessed_lemma"):
            Token.set_extension("preprocessed_lemma", default=None)

    def _normalize_text(self, text: str) -> str:
        """Apply the full normalization chain to a string."""
        if self._long_s_normalizer is not None:
            text = self._long_s_normalizer.normalize_text_full(
                text, apply_pass2=self.long_s_pass2
            )
        if self._uv_normalizer is not None:
            text = self._uv_normalizer.normalize(text)
        return text

    def _normalize_word(self, word: str) -> str:
        """Apply the full normalization chain to a single word."""
        if self._long_s_normalizer is not None:
            word, _ = self._long_s_normalizer.normalize_word_full(
                word, apply_pass2=self.long_s_pass2
            )
        if self._uv_normalizer is not None:
            word = self._uv_normalizer.normalize(word)
        return word

    def __call__(self, doc: Doc) -> Doc:
        doc._.preprocessed = self._normalize_text(doc.text)

        for token in doc:
            token._.preprocessed = self._normalize_word(token.text)
            token._.preprocessed_lemma = self._normalize_word(token.lemma_)

        return doc

    def to_disk(self, path: str, *, exclude: tuple[str, ...] = ()) -> None:
        pass

    def from_disk(
        self, path: str, *, exclude: tuple[str, ...] = ()
    ) -> "LatinPreprocessorComponent":
        return self

    def to_bytes(self, *, exclude: tuple[str, ...] = ()) -> bytes:
        return b""

    def from_bytes(
        self, data: bytes, *, exclude: tuple[str, ...] = ()
    ) -> "LatinPreprocessorComponent":
        return self


# =============================================================================
# U/V Normalizer Component
# =============================================================================


@Language.factory(
    "uv_normalizer",
    default_config={"method": "rules"},
    assigns=["doc._.uv_normalized", "token._.uv_normalized", "token._.uv_normalized_lemma"],
)
def create_uv_normalizer(
    nlp: Language,
    name: str,
    method: str = "rules",
) -> "UVNormalizerComponent":
    """Create a U/V normalizer pipeline component."""
    return UVNormalizerComponent(nlp, name, method=method)


class UVNormalizerComponent:
    """
    spaCy pipeline component for U/V normalization.

    Extensions:
        - Doc._.uv_normalized: Full normalized text.
        - Token._.uv_normalized: Normalized token text.
        - Token._.uv_normalized_lemma: V-space lemma (u→v normalized).

    Note: token.lemma_ is NEVER modified. Lemmas stay in u-space internally
    to match training data. Use token._.uv_normalized_lemma for v-space lemmas.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        method: str = "rules",
    ) -> None:
        self.name = name
        self.method = method

        if method != "rules":
            raise ValueError(
                f"Unknown method: {method}. Currently only 'rules' is supported."
            )

        self._normalizer = UVNormalizerRules()

        if not Doc.has_extension("uv_normalized"):
            Doc.set_extension("uv_normalized", default=None)
        if not Token.has_extension("uv_normalized"):
            Token.set_extension("uv_normalized", default=None)
        if not Token.has_extension("uv_normalized_lemma"):
            Token.set_extension("uv_normalized_lemma", default=None)

    def __call__(self, doc: Doc) -> Doc:
        doc._.uv_normalized = self._normalizer.normalize(doc.text)

        for token in doc:
            token._.uv_normalized = self._normalizer.normalize(token.text)
            token._.uv_normalized_lemma = self._normalizer.normalize(token.lemma_)

        return doc

    def to_disk(self, path: str, *, exclude: tuple[str, ...] = ()) -> None:
        pass

    def from_disk(
        self, path: str, *, exclude: tuple[str, ...] = ()
    ) -> "UVNormalizerComponent":
        return self

    def to_bytes(self, *, exclude: tuple[str, ...] = ()) -> bytes:
        return b""

    def from_bytes(
        self, data: bytes, *, exclude: tuple[str, ...] = ()
    ) -> "UVNormalizerComponent":
        return self


# =============================================================================
# Long-S Normalizer Component
# =============================================================================


@Language.factory(
    "long_s_normalizer",
    default_config={"apply_pass2": True},
    assigns=["doc._.long_s_normalized", "token._.long_s_normalized"],
)
def create_long_s_normalizer(
    nlp: Language,
    name: str,
    apply_pass2: bool = True,
) -> "LongSNormalizerComponent":
    """Create a long-s normalizer pipeline component."""
    return LongSNormalizerComponent(nlp, name, apply_pass2=apply_pass2)


class LongSNormalizerComponent:
    """
    spaCy pipeline component for long-s OCR correction.

    Extensions:
        - Doc._.long_s_normalized: Full normalized text.
        - Token._.long_s_normalized: Normalized token text.
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        apply_pass2: bool = True,
    ) -> None:
        self.name = name
        self.apply_pass2 = apply_pass2
        self._normalizer = LongSNormalizer()

        if not Doc.has_extension("long_s_normalized"):
            Doc.set_extension("long_s_normalized", default=None)
        if not Token.has_extension("long_s_normalized"):
            Token.set_extension("long_s_normalized", default=None)

    def __call__(self, doc: Doc) -> Doc:
        doc._.long_s_normalized = self._normalizer.normalize_text_full(
            doc.text, apply_pass2=self.apply_pass2
        )

        for token in doc:
            normalized, _ = self._normalizer.normalize_word_full(
                token.text, apply_pass2=self.apply_pass2
            )
            token._.long_s_normalized = normalized

        return doc

    def to_disk(self, path: str, *, exclude: tuple[str, ...] = ()) -> None:
        pass

    def from_disk(
        self, path: str, *, exclude: tuple[str, ...] = ()
    ) -> "LongSNormalizerComponent":
        return self

    def to_bytes(self, *, exclude: tuple[str, ...] = ()) -> bytes:
        return b""

    def from_bytes(
        self, data: bytes, *, exclude: tuple[str, ...] = ()
    ) -> "LongSNormalizerComponent":
        return self


# =============================================================================
# Utility Functions
# =============================================================================


def get_normalizer_pipe(nlp: Language) -> Optional[UVNormalizerComponent]:
    """Get the U/V normalizer component from a pipeline."""
    if "uv_normalizer" in nlp.pipe_names:
        return nlp.get_pipe("uv_normalizer")
    return None
