"""Tests for spaCy pipeline components."""

import pytest
import spacy


@pytest.fixture(autouse=True)
def _clean_extensions():
    """Remove custom extensions between tests to avoid conflicts."""
    from spacy.tokens import Doc, Token

    yield

    for ext in [
        "preprocessed", "uv_normalized", "long_s_normalized",
    ]:
        if Doc.has_extension(ext):
            Doc.remove_extension(ext)

    for ext in [
        "preprocessed", "preprocessed_lemma",
        "uv_normalized", "uv_normalized_lemma",
        "long_s_normalized",
    ]:
        if Token.has_extension(ext):
            Token.remove_extension(ext)


class TestLatinPreprocessor:
    """Test the unified latin_preprocessor component."""

    def test_factory_registered(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        assert "latin_preprocessor" in nlp.pipe_names

    def test_uv_normalization(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        doc = nlp("uirumque")
        assert doc[0]._.preprocessed == "virumque"

    def test_long_s_normalization(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        # Long-s preserves case
        doc = nlp("Arma")
        assert doc[0]._.preprocessed == "Arma"

    def test_combined_long_s_and_uv(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        # Long-s runs first (preserves case), then U/V applies
        doc = nlp("Arma uirumque cano")
        assert doc._.preprocessed == "Arma virumque cano"

    def test_token_preprocessed_lemma(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        doc = nlp("uirumque")
        assert doc[0]._.preprocessed_lemma is not None

    def test_disable_long_s(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor", config={"long_s": False})
        doc = nlp("ftatua")
        # long-s disabled, U/V still applies: ftatua → ftatva
        assert doc[0]._.preprocessed == "ftatva"

    def test_disable_uv(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor", config={"uv": False})
        doc = nlp("uirumque")
        # u/v disabled, so "u" stays
        assert doc[0]._.preprocessed == "uirumque"

    def test_disable_both(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor", config={"long_s": False, "uv": False})
        doc = nlp("ftatua uirumque")
        assert doc[0]._.preprocessed == "ftatua"
        assert doc[1]._.preprocessed == "uirumque"

    def test_passthrough_clean_text(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        # Case is preserved through normalization
        doc = nlp("Caesar in Galliam contendit")
        assert doc._.preprocessed == "Caesar in Galliam contendit"

    def test_serialization_roundtrip(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("latin_preprocessor")
        pipe = nlp.get_pipe("latin_preprocessor")
        data = pipe.to_bytes()
        pipe.from_bytes(data)


class TestUVNormalizerComponent:
    """Test the standalone uv_normalizer component."""

    def test_factory_registered(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("uv_normalizer")
        assert "uv_normalizer" in nlp.pipe_names

    def test_basic_normalization(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("uv_normalizer")
        doc = nlp("Arma uirumque cano")
        assert doc._.uv_normalized == "Arma virumque cano"

    def test_token_extensions(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("uv_normalizer")
        doc = nlp("uirumque")
        assert doc[0]._.uv_normalized == "virumque"


class TestLongSNormalizerComponent:
    """Test the standalone long_s_normalizer component."""

    def test_factory_registered(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("long_s_normalizer")
        assert "long_s_normalizer" in nlp.pipe_names

    def test_basic_normalization(self):
        nlp = spacy.blank("la")
        nlp.add_pipe("long_s_normalizer")
        doc = nlp("ftatua")
        assert doc[0]._.long_s_normalized == "statua"
