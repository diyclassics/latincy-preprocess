"""
Backend detection and Rust/Python parity tests for both normalizers.
"""

import pytest

from latincy_preprocess import backend
from latincy_preprocess.uv import (
    Change,
    NormalizationResult,
    UVNormalizerRules,
    normalize_uv,
)
from latincy_preprocess.uv._rules import (
    UVNormalizerRules as PythonUVNormalizer,
    normalize_uv as python_normalize_uv,
)
from latincy_preprocess.long_s import LongSNormalizer
from latincy_preprocess.long_s._rules import LongSNormalizer as PythonLongSNormalizer


# =============================================================================
# Backend Detection
# =============================================================================


class TestBackendDetection:
    def test_backend_returns_valid_string(self):
        assert backend() in ("rust", "python")

    def test_python_uv_always_available(self):
        normalizer = PythonUVNormalizer()
        assert normalizer.normalize("uia") == "via"

    def test_python_long_s_always_available(self):
        normalizer = PythonLongSNormalizer()
        result, _ = normalizer.normalize_word_full("ftatua")
        assert result == "statua"


# =============================================================================
# UV Parity Helpers
# =============================================================================


@pytest.fixture
def rust_available():
    if backend() != "rust":
        pytest.skip("Rust backend not available")


@pytest.fixture
def active_uv_normalizer():
    return UVNormalizerRules()


@pytest.fixture
def python_uv_normalizer():
    return PythonUVNormalizer()


def assert_uv_parity(word, py, active):
    py_result = py.normalize(word)
    active_result = active.normalize(word)
    assert py_result == active_result, (
        f"UV parity failure for {word!r}: "
        f"python={py_result!r}, active={active_result!r}"
    )


# =============================================================================
# UV Parity Tests
# =============================================================================


class TestUVParity:
    @pytest.mark.parametrize("word", [
        "quod", "aqua", "quid", "quae", "quinque", "sequitur",
    ])
    def test_qu_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "lingua", "sanguis", "pinguis", "unguis",
    ])
    def test_ngu_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "cui", "cuius", "huic", "sua", "tua", "duo", "eius", "perpetuum",
    ])
    def test_word_exception_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "fuit", "fui", "potuit", "tenuit", "habuit",
        "fuisse", "fuerat", "fuere", "voluit", "noluit",
    ])
    def test_perfect_tense_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "seruus", "fluuius", "nouus", "iuuat", "paruus",
    ])
    def test_double_u_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "uia", "uir", "uox", "uinum", "uerus",
    ])
    def test_initial_before_vowel_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("word", [
        "nouo", "breuis", "auis", "caueo", "moueo",
    ])
    def test_intervocalic_parity(self, word, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(word, python_uv_normalizer, active_uv_normalizer)

    @pytest.mark.parametrize("sentence", [
        "Arma uirumque cano Troiae qui primus ab oris",
        "Gallia est omnis diuisa in partes tres",
        "SENATVS POPVLVSQVE ROMANVS",
    ])
    def test_sentence_parity(self, sentence, python_uv_normalizer, active_uv_normalizer, rust_available):
        assert_uv_parity(sentence, python_uv_normalizer, active_uv_normalizer)


# =============================================================================
# Long-S Parity
# =============================================================================


class TestLongSParity:
    @pytest.fixture
    def py_normalizer(self):
        return PythonLongSNormalizer()

    @pytest.fixture
    def has_rust(self):
        try:
            from latincy_preprocess import _rust  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Rust backend not available")

    WORDS = [
        "ftatua", "fpiritus", "fcilicet", "fquam", "fpecies",
        "fufpiciens", "fumma", "funt", "fed", "fecit", "fuit",
        "dominus", "rex", "", "f", "fff", "FTATUA", "123",
        "feipfum", "teipfum", "chriftus", "noftra", "ipfum",
        "ef", "poteft", "fenatuf",
    ]

    @pytest.mark.parametrize("word", WORDS)
    def test_pass1_parity(self, py_normalizer, has_rust, word):
        from latincy_preprocess._rust import normalize_long_s_word_pass1
        py_result, _ = py_normalizer.normalize_word_pass1(word)
        rust_result = normalize_long_s_word_pass1(word)
        assert rust_result == py_result, (
            f"Parity failure on {word!r}: python={py_result!r}, rust={rust_result!r}"
        )

    @pytest.mark.parametrize("word", WORDS)
    def test_full_parity(self, py_normalizer, has_rust, word):
        from latincy_preprocess._rust import normalize_long_s_word_full
        py_result, _ = py_normalizer.normalize_word_full(word, apply_pass2=True)
        rust_result = normalize_long_s_word_full(word, True)
        assert rust_result == py_result, (
            f"Parity failure on {word!r}: python={py_result!r}, rust={rust_result!r}"
        )

    def test_text_parity(self, py_normalizer, has_rust):
        from latincy_preprocess._rust import normalize_long_s_text_full
        text = "funt ftatua fundamentum fpiritus chriftus noftra"
        py_result = py_normalizer.normalize_text_full(text, apply_pass2=True)
        rust_result = normalize_long_s_text_full(text, True)
        assert rust_result == py_result
