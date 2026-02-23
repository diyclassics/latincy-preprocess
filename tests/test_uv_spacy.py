"""
Tests for the spaCy U/V normalizer integration.

Requires spaCy to be installed.
"""

import pytest

spacy = pytest.importorskip("spacy")

from latincy_preprocess.spacy import UVNormalizerComponent, create_uv_normalizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nlp():
    return spacy.blank("la")


@pytest.fixture
def nlp_with_normalizer(nlp):
    nlp.add_pipe("uv_normalizer")
    return nlp


@pytest.fixture
def nlp_with_lemma_normalizer(nlp):
    from spacy.language import Language
    from spacy.tokens import Doc

    @Language.component("mock_lemmatizer")
    def mock_lemmatizer(doc: Doc) -> Doc:
        for token in doc:
            token.lemma_ = token.text
        return doc

    nlp.add_pipe("mock_lemmatizer")
    nlp.add_pipe("uv_normalizer")
    return nlp


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    def test_factory_registered(self, nlp):
        assert "uv_normalizer" in spacy.registry.factories

    def test_add_pipe(self, nlp):
        nlp.add_pipe("uv_normalizer")
        assert "uv_normalizer" in nlp.pipe_names

    def test_factory_returns_component(self, nlp):
        pipe = nlp.add_pipe("uv_normalizer")
        assert isinstance(pipe, UVNormalizerComponent)

    def test_invalid_method(self, nlp):
        with pytest.raises(ValueError, match="Unknown method"):
            nlp.add_pipe("uv_normalizer", config={"method": "invalid"})


# =============================================================================
# Component Tests
# =============================================================================


class TestComponent:
    def test_component_name(self, nlp_with_normalizer):
        pipe = nlp_with_normalizer.get_pipe("uv_normalizer")
        assert pipe.name == "uv_normalizer"

    def test_component_method(self, nlp_with_normalizer):
        pipe = nlp_with_normalizer.get_pipe("uv_normalizer")
        assert pipe.method == "rules"


# =============================================================================
# Extension Tests
# =============================================================================


class TestExtensions:
    def test_doc_extension_registered(self, nlp_with_normalizer):
        from spacy.tokens import Doc
        assert Doc.has_extension("uv_normalized")

    def test_token_extension_registered(self, nlp_with_normalizer):
        from spacy.tokens import Token
        assert Token.has_extension("uv_normalized")

    def test_token_lemma_extension_registered(self, nlp_with_normalizer):
        from spacy.tokens import Token
        assert Token.has_extension("uv_normalized_lemma")

    def test_doc_normalized(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("Arma uirumque cano")
        assert doc._.uv_normalized == "Arma virumque cano"

    def test_token_normalized(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("uirumque")
        assert doc[0]._.uv_normalized == "virumque"

    def test_multiple_tokens(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("uia uita uox")
        assert doc[0]._.uv_normalized == "via"
        assert doc[1]._.uv_normalized == "vita"
        assert doc[2]._.uv_normalized == "vox"


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    def test_classic_example(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("Arma uirumque cano")
        assert doc._.uv_normalized == "Arma virumque cano"

    def test_caesar_quote(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("Veni, uidi, uici")
        assert doc._.uv_normalized == "Veni, vidi, vici"

    def test_qu_preserved(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("quod quid quae")
        assert doc._.uv_normalized == "quod quid quae"

    def test_perfect_tense(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("fuit potuit habuit")
        assert doc._.uv_normalized == "fuit potuit habuit"

    def test_case_preserved(self, nlp_with_normalizer):
        doc = nlp_with_normalizer("VIA Via via")
        assert doc._.uv_normalized == "VIA Via via"


# =============================================================================
# Lemma Normalization Tests
# =============================================================================


class TestLemmaNormalization:
    """Test that token.lemma_ is NOT modified (stays u-space) and
    token._.uv_normalized_lemma provides v-space lemmas."""

    def test_lemma_not_modified(self, nlp_with_lemma_normalizer):
        """token.lemma_ must stay in u-space (unchanged from upstream)."""
        doc = nlp_with_lemma_normalizer("uiuo")
        # mock_lemmatizer sets lemma_ = token.text, uv_normalizer must NOT change it
        assert doc[0].lemma_ == "uiuo"

    def test_extension_has_vspace(self, nlp_with_lemma_normalizer):
        """token._.uv_normalized_lemma has the v-space version."""
        doc = nlp_with_lemma_normalizer("uiuo")
        assert doc[0]._.uv_normalized_lemma == "vivo"

    def test_venio(self, nlp_with_lemma_normalizer):
        doc = nlp_with_lemma_normalizer("uenio")
        assert doc[0].lemma_ == "uenio"  # u-space preserved
        assert doc[0]._.uv_normalized_lemma == "venio"

    def test_no_change(self, nlp_with_lemma_normalizer):
        doc = nlp_with_lemma_normalizer("sum")
        assert doc[0].lemma_ == "sum"
        assert doc[0]._.uv_normalized_lemma == "sum"

    def test_qu_preserved(self, nlp_with_lemma_normalizer):
        doc = nlp_with_lemma_normalizer("quod")
        assert doc[0].lemma_ == "quod"
        assert doc[0]._.uv_normalized_lemma == "quod"

    def test_perfect_u_preserved(self, nlp_with_lemma_normalizer):
        doc = nlp_with_lemma_normalizer("fuit")
        assert doc[0].lemma_ == "fuit"
        assert doc[0]._.uv_normalized_lemma == "fuit"

    def test_multi_token_lemmas(self, nlp_with_lemma_normalizer):
        doc = nlp_with_lemma_normalizer("uiuo uenio fuit quod")
        # lemma_ stays u-space
        assert doc[0].lemma_ == "uiuo"
        assert doc[1].lemma_ == "uenio"
        assert doc[2].lemma_ == "fuit"
        assert doc[3].lemma_ == "quod"
        # extensions have v-space
        assert doc[0]._.uv_normalized_lemma == "vivo"
        assert doc[1]._.uv_normalized_lemma == "venio"
        assert doc[2]._.uv_normalized_lemma == "fuit"
        assert doc[3]._.uv_normalized_lemma == "quod"


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    def test_to_bytes(self, nlp_with_normalizer):
        pipe = nlp_with_normalizer.get_pipe("uv_normalizer")
        data = pipe.to_bytes()
        assert isinstance(data, bytes)

    def test_from_bytes(self, nlp_with_normalizer):
        pipe = nlp_with_normalizer.get_pipe("uv_normalizer")
        data = pipe.to_bytes()
        pipe2 = pipe.from_bytes(data)
        assert isinstance(pipe2, UVNormalizerComponent)

    def test_pipeline_to_disk(self, nlp_with_normalizer, tmp_path):
        nlp_with_normalizer.to_disk(tmp_path / "model")
        assert (tmp_path / "model").exists()

    def test_pipeline_from_disk(self, nlp_with_normalizer, tmp_path):
        nlp_with_normalizer.to_disk(tmp_path / "model")
        nlp2 = spacy.load(tmp_path / "model")
        assert "uv_normalizer" in nlp2.pipe_names
        doc = nlp2("uia")
        assert doc._.uv_normalized == "via"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_with_sentencizer(self, nlp):
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("uv_normalizer")
        doc = nlp("Arma uirumque cano. Veni uidi uici.")
        assert doc._.uv_normalized == "Arma virumque cano. Veni vidi vici."
        assert len(list(doc.sents)) == 2

    def test_pipe_order(self, nlp):
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("uv_normalizer", first=True)
        assert nlp.pipe_names[0] == "uv_normalizer"

    def test_batch_processing(self, nlp_with_normalizer):
        texts = ["uia", "uita", "uox", "uinum"]
        docs = list(nlp_with_normalizer.pipe(texts))
        assert [doc._.uv_normalized for doc in docs] == ["via", "vita", "vox", "vinum"]
