"""
Tests for Long-S normalization.

Generated from manually validated data:
- 54 approved corrections (100% accuracy)
- 107 allowlisted words (legitimate f- words that must be preserved)
"""

import json
from pathlib import Path

import pytest

from latincy_preprocess.long_s import LongSNormalizer

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normalizer():
    """Fresh LongSNormalizer instance for each test."""
    return LongSNormalizer()


@pytest.fixture
def approved_corrections():
    """Load the 54 manually approved corrections."""
    path = DATA_DIR / "approved_corrections.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def allowlist():
    """Load the allowlist of legitimate f- words."""
    path = DATA_DIR / "allowlist.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Section 1: Approved corrections -- the normalizer must reproduce all 54
# ===========================================================================

def _load_approved_corrections():
    path = DATA_DIR / "approved_corrections.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for key in data:
        ocr_form, expected = key.split("\u2192")
        pairs.append((ocr_form, expected))
    return pairs


@pytest.mark.parametrize(
    "ocr_form,expected",
    _load_approved_corrections(),
    ids=[f"{ocr}->{exp}" for ocr, exp in _load_approved_corrections()],
)
def test_approved_correction(normalizer, ocr_form, expected):
    result, _rules = normalizer.normalize_word_full(ocr_form, apply_pass2=True)
    assert result == expected, (
        f"Approved correction failed: {ocr_form!r} -> expected {expected!r}, got {result!r}"
    )


# ===========================================================================
# Section 2: Allowlist preservation
# ===========================================================================

def _load_allowlist():
    path = DATA_DIR / "allowlist.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("word", _load_allowlist())
def test_allowlist_preserved(normalizer, word):
    result, _rules = normalizer.normalize_word_full(word, apply_pass2=True)
    assert result == word, (
        f"Allowlisted word was incorrectly modified: {word!r} -> {result!r}"
    )


# ===========================================================================
# Section 3: Pass 1 high-confidence rules
# ===========================================================================

class TestPass1Rules:
    def test_ft_to_st(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("ftatua")
        assert result == "statua"
        assert any("ft" in r for r in rules)

    def test_fp_to_sp(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fpiritus")
        assert result == "spiritus"
        assert any("fp" in r for r in rules)

    def test_fc_to_sc(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fcilicet")
        assert result == "scilicet"
        assert any("fc" in r for r in rules)

    def test_fqu_to_squ(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fquam")
        assert result == "squam"
        assert any("fqu" in r for r in rules)

    def test_fpe_to_spe(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fpecies")
        assert result == "species"
        assert any("fpe" in r for r in rules)

    def test_fuf_to_sus(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fufpiciens")
        assert result == "suspiciens"

    def test_fum_to_sum(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fumma")
        assert result == "summa"
        assert any("fum" in r for r in rules)

    def test_multiple_rules_apply(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("tranfponere")
        assert result == "transponere"

    def test_medial_fp(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("profpera")
        assert result == "prospera"

    def test_no_false_positive_on_clean_word(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("dominus")
        assert result == "dominus"
        assert rules == []


# ===========================================================================
# Section 4: Word-final f -> s
# ===========================================================================

class TestWordFinalF:
    def test_word_final_f_simple(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("ef")
        assert result == "es"
        assert any("word-final" in r for r in rules)

    def test_word_final_f_longer_word(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("poteft")
        assert result == "potest"

    def test_word_final_f_with_prefix(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fenatuf")
        assert result == "fenatus"

    def test_single_f(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("f")
        assert result == "s"

    def test_word_ending_in_non_f(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("rex")
        assert result == "rex"
        assert not any("word-final" in r for r in rules)


# ===========================================================================
# Section 5: Pass 2 context-dependent rules
# ===========================================================================

class TestPass2Rules:
    def test_fu_to_su_common(self, normalizer):
        result, rules = normalizer.normalize_word_full("funt", apply_pass2=True)
        assert result == "sunt", f"Expected 'sunt', got {result!r}"

    def test_fe_to_se_common(self, normalizer):
        result, rules = normalizer.normalize_word_full("fed", apply_pass2=True)
        assert result == "sed", f"Expected 'sed', got {result!r}"

    def test_allowlist_blocks_pass2(self, normalizer):
        result, rules = normalizer.normalize_word_full("fuit", apply_pass2=True)
        assert result == "fuit"

    def test_allowlist_fe_blocks_pass2(self, normalizer):
        result, rules = normalizer.normalize_word_full("fecit", apply_pass2=True)
        assert result == "fecit"

    def test_pass2_disabled(self, normalizer):
        result, rules = normalizer.normalize_word_full("funt", apply_pass2=False)
        assert result == "funt"


# ===========================================================================
# Section 6: Text-level normalization
# ===========================================================================

class TestTextNormalization:
    def test_text_pass1(self, normalizer):
        text = "ftatua fpiritus fcilicet"
        result = normalizer.normalize_text_pass1(text)
        assert result == "statua spiritus scilicet"

    def test_text_full(self, normalizer):
        text = "funt ftatua fundamentum"
        result = normalizer.normalize_text_full(text, apply_pass2=True)
        assert result == "sunt statua fundamentum"

    def test_empty_text(self, normalizer):
        result = normalizer.normalize_text_full("")
        assert result == ""

    def test_single_word_text(self, normalizer):
        result = normalizer.normalize_text_full("fpiritus")
        assert result == "spiritus"

    def test_preserves_spacing(self, normalizer):
        result = normalizer.normalize_text_full("rex   dominus")
        assert result == "rex dominus"

    def test_text_case_preservation(self, normalizer):
        result = normalizer.normalize_text_full("Sic uita eft", apply_pass2=True)
        assert result == "Sic uita est"

    def test_text_allcaps_preservation(self, normalizer):
        result = normalizer.normalize_text_full("FTATUA FPIRITUS", apply_pass2=True)
        assert result == "STATUA SPIRITUS"


# ===========================================================================
# Section 7: Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    def test_empty_word_pass1(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("")
        assert result == ""
        assert rules == []

    def test_empty_word_full(self, normalizer):
        result, rules = normalizer.normalize_word_full("")
        assert result == ""

    def test_single_character_non_f(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("a")
        assert result == "a"

    def test_all_f_word(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("fff")
        assert result == "ffs"

    def test_uppercase_preserved(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("FTATUA")
        assert result == "STATUA"

    def test_mixed_case_title(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("Fpiritus")
        assert result == "Spiritus"

    def test_lowercase_unchanged(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("ftatua")
        assert result == "statua"

    def test_numeric_input(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("123")
        assert result == "123"
        assert rules == []

    def test_punctuation_attached(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("eft,")
        assert "st" in result

    def test_non_latin_characters(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("\u03b1\u03b2\u03b3")
        assert result == "\u03b1\u03b2\u03b3"
        assert rules == []

    def test_actual_long_s_character(self, normalizer):
        result, rules = normalizer.normalize_word_pass1("\u017ftatua")
        assert result == "\u017ftatua"


# ===========================================================================
# Section 8: Statistics tracking
# ===========================================================================

class TestStatistics:
    def test_stats_count_words(self, normalizer):
        normalizer.normalize_text_pass1("eft fpiritus rex")
        assert normalizer.stats["total_words"] == 3

    def test_stats_count_modified(self, normalizer):
        normalizer.normalize_text_pass1("eft fpiritus rex")
        assert normalizer.stats["words_modified"] == 2

    def test_stats_reset(self, normalizer):
        normalizer.normalize_text_pass1("eft fpiritus")
        normalizer.reset_statistics()
        assert normalizer.stats["total_words"] == 0
        assert normalizer.stats["words_modified"] == 0
        assert normalizer.stats["transformations"] == {}

    def test_stats_track_transformations(self, normalizer):
        normalizer.normalize_word_pass1("ftatua")
        assert "ft \u2192 st" in normalizer.stats["transformations"]


# ===========================================================================
# Section 9: Regression tests from known corpus examples
# ===========================================================================

class TestCorpusRegressions:
    def test_christus_family(self, normalizer):
        cases = {
            "chriftus": "christus",
            "chriftum": "christum",
            "chrifti": "christi",
            "chrifto": "christo",
            "chriftiani": "christiani",
        }
        for ocr, expected in cases.items():
            result, _ = normalizer.normalize_word_pass1(ocr)
            assert result == expected, f"{ocr} -> expected {expected}, got {result}"

    def test_noster_family(self, normalizer):
        cases = {
            "noftra": "nostra",
            "noftram": "nostram",
            "noftro": "nostro",
            "noftre": "nostre",
            "noftris": "nostris",
            "noftri": "nostri",
            "noftrum": "nostrum",
        }
        for ocr, expected in cases.items():
            result, _ = normalizer.normalize_word_pass1(ocr)
            assert result == expected, f"{ocr} -> expected {expected}, got {result}"

    def test_ipsum_family(self, normalizer):
        result, _ = normalizer.normalize_word_pass1("ipfum")
        assert result == "ipsum"

    def test_spiritus_family(self, normalizer):
        cases = {
            "fpiritus": "spiritus",
            "fpiritualis": "spiritualis",
            "fpiritu": "spiritu",
        }
        for ocr, expected in cases.items():
            result, _ = normalizer.normalize_word_pass1(ocr)
            assert result == expected, f"{ocr} -> expected {expected}, got {result}"

    def test_compound_corrections(self, normalizer):
        result, _ = normalizer.normalize_word_full("feipfum", apply_pass2=True)
        assert result == "seipsum"

        result, _ = normalizer.normalize_word_full("teipfum", apply_pass2=True)
        assert result == "teipsum"
