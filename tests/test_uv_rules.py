"""
Tests for the rule-based U/V normalizer.

Runs against the curated test set (100 cases) covering all rule categories.
"""

import json
from pathlib import Path

import pytest

from latincy_preprocess.uv import Change, NormalizationResult, UVNormalizerRules, normalize_uv


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def normalizer() -> UVNormalizerRules:
    """Return a fresh normalizer instance."""
    return UVNormalizerRules()


@pytest.fixture
def curated_tests() -> list[dict]:
    """Load curated test cases from JSON file."""
    test_file = Path(__file__).parent / "data" / "curated_test.json"
    with open(test_file) as f:
        data = json.load(f)
    return data["test_cases"]


# =============================================================================
# Basic API Tests
# =============================================================================


class TestBasicAPI:
    """Test basic API functionality."""

    def test_empty_string(self, normalizer: UVNormalizerRules):
        assert normalizer.normalize("") == ""

    def test_no_uv_characters(self, normalizer: UVNormalizerRules):
        text = "Roma aeterna est"
        assert normalizer.normalize(text) == text

    def test_preserves_case_lowercase(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize("uia")
        assert result == "via"
        assert result[0].islower()

    def test_preserves_case_uppercase(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize("VIA")
        assert result == "VIA"
        assert result[0].isupper()

    def test_mixed_case(self, normalizer: UVNormalizerRules):
        assert normalizer.normalize("Veni VIDI uici") == "Veni VIDI vici"


class TestModuleFunction:
    def test_normalize_uv_function(self):
        assert normalize_uv("uia") == "via"
        assert normalize_uv("Arma uirumque cano") == "Arma virumque cano"

    def test_normalize_uv_singleton(self):
        result1 = normalize_uv("test")
        result2 = normalize_uv("test")
        assert result1 == result2


class TestNormalizeChar:
    def test_normalize_char_returns_tuple(self, normalizer: UVNormalizerRules):
        char, rule = normalizer.normalize_char("uia", 0)
        assert char == "v"
        assert rule == "initial_before_vowel"

    def test_normalize_char_preserves_case(self, normalizer: UVNormalizerRules):
        char, _ = normalizer.normalize_char("VIA", 0)
        assert char == "V"

    def test_normalize_char_invalid_position(self, normalizer: UVNormalizerRules):
        with pytest.raises(ValueError, match="not u/v"):
            normalizer.normalize_char("abc", 0)


class TestNormalizeDetailed:
    def test_returns_normalization_result(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize_detailed("uia")
        assert isinstance(result, NormalizationResult)

    def test_result_fields(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize_detailed("uia")
        assert result.original == "uia"
        assert result.normalized == "via"
        assert len(result.changes) == 1

    def test_change_details(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize_detailed("uia")
        change = result.changes[0]
        assert isinstance(change, Change)
        assert change.position == 0
        assert change.original == "u"
        assert change.normalized == "v"
        assert change.rule == "initial_before_vowel"

    def test_no_changes_recorded_when_identical(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize_detailed("quod")
        changes_at_u = [c for c in result.changes if c.position == 1]
        assert len(changes_at_u) == 0

    def test_empty_input_detailed(self, normalizer: UVNormalizerRules):
        result = normalizer.normalize_detailed("")
        assert result.original == ""
        assert result.normalized == ""
        assert result.changes == []


# =============================================================================
# Rule Category Tests
# =============================================================================


class TestQuDigraph:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("quod", "quod"),
            ("aqua", "aqua"),
            ("quid", "quid"),
            ("quae", "quae"),
            ("quinque", "quinque"),
            ("sequitur", "sequitur"),
        ],
    )
    def test_qu_preserved(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestNguDigraph:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("lingua", "lingua"),
            ("sanguis", "sanguis"),
            ("pinguis", "pinguis"),
            ("unguis", "unguis"),
        ],
    )
    def test_ngu_preserved(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestWordExceptions:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("cui", "cui"),
            ("cuius", "cuius"),
            ("huic", "huic"),
            ("sua", "sua"),
            ("tua", "tua"),
            ("duo", "duo"),
            ("eius", "eius"),
            ("perpetuum", "perpetuum"),
        ],
    )
    def test_word_exceptions(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestPerfectTense:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("fuit", "fuit"),
            ("fui", "fui"),
            ("potuit", "potuit"),
            ("tenuit", "tenuit"),
            ("habuit", "habuit"),
            ("fuisse", "fuisse"),
            ("fuerat", "fuerat"),
            ("fuere", "fuere"),
            ("voluit", "voluit"),
            ("noluit", "noluit"),
        ],
    )
    def test_u_perfect_preserved(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected

    def test_v_perfect_distinguished(self, normalizer):
        assert normalizer.normalize("soluit") == "solvit"


class TestDoubleU:
    @pytest.mark.parametrize(
        "input_text,expected,description",
        [
            ("seruus", "servus", "V-C-uu: first v, second u"),
            ("fluuius", "fluvius", "C-C-uu: first u, second v"),
            ("nouus", "novus", "V-uu: first v, second u"),
            ("iuuat", "iuvat", "initial i-uu: first u, second v"),
            ("paruus", "parvus", "V-C-uu with r"),
        ],
    )
    def test_double_u_patterns(self, normalizer, input_text, expected, description):
        assert normalizer.normalize(input_text) == expected, description


class TestInitialPosition:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("uia", "via"),
            ("uir", "vir"),
            ("uox", "vox"),
            ("uinum", "vinum"),
            ("uerus", "verus"),
        ],
    )
    def test_initial_before_vowel(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("urbs", "urbs"),
            ("usus", "usus"),
            ("unda", "unda"),
            ("unus", "unus"),
        ],
    )
    def test_initial_before_consonant(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestIntervocalic:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("nouo", "novo"),
            ("breuis", "brevis"),
            ("auis", "avis"),
            ("caueo", "caveo"),
            ("moueo", "moveo"),
        ],
    )
    def test_intervocalic_becomes_v(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestBeforeConsonant:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("scriptum", "scriptum"),
            ("Augustus", "Augustus"),
            ("causa", "causa"),
            ("aurum", "aurum"),
            ("laudat", "laudat"),
        ],
    )
    def test_before_consonant_stays_u(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestWordFinal:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("tu", "tu"),
            ("cum", "cum"),
            ("dum", "dum"),
            ("sum", "sum"),
        ],
    )
    def test_word_final_stays_u(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestPostConsonant:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("silua", "silva"),
            ("seruo", "servo"),
            ("soluo", "solvo"),
        ],
    )
    def test_post_consonant_before_vowel(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("cultus", "cultus"),
        ],
    )
    def test_post_consonant_before_consonant(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


class TestVocalicUStems:
    """Test that vocalic-u stems prevent false v classification."""

    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("statua", "statua"),
            ("statuae", "statuae"),
            ("statuam", "statuam"),
            ("statuas", "statuas"),
            ("ardua", "ardua"),
            ("arduo", "arduo"),
            ("fatua", "fatua"),
            ("residua", "residua"),
            ("strenua", "strenua"),
            ("conspicua", "conspicua"),
            ("individua", "individua"),
        ],
    )
    def test_vocalic_u_stem_words(self, normalizer, input_text, expected):
        assert normalizer.normalize(input_text) == expected


# =============================================================================
# Curated Test Set
# =============================================================================


class TestCuratedSet:
    def test_curated_test_count(self, curated_tests):
        assert len(curated_tests) == 100

    def test_all_curated_cases(self, normalizer, curated_tests):
        failures = []
        for case in curated_tests:
            result = normalizer.normalize(case["input"])
            if result != case["expected"]:
                failures.append(
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "input": case["input"],
                        "expected": case["expected"],
                        "got": result,
                        "description": case.get("description", ""),
                    }
                )

        if failures:
            msg = f"\n{len(failures)} test case(s) failed:\n"
            for f in failures:
                msg += (
                    f"  #{f['id']} [{f['category']}]: {f['input']!r} -> "
                    f"expected {f['expected']!r}, got {f['got']!r}\n"
                    f"    ({f['description']})\n"
                )
            pytest.fail(msg)

    @pytest.mark.parametrize("category", [
        "qu_digraph",
        "gu_ngu_digraph",
        "word_exceptions",
        "perfect_tense",
        "double_u",
        "initial_position",
        "intervocalic",
        "before_consonant",
        "word_final",
        "post_consonant",
        "mixed_complex",
    ])
    def test_category_coverage(self, curated_tests, category):
        category_cases = [c for c in curated_tests if c["category"] == category]
        assert len(category_cases) > 0, f"No test cases for category {category}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_all_uppercase_spqr(self, normalizer):
        assert normalizer.normalize("SENATVS POPVLVSQVE ROMANVS") == \
               "SENATUS POPULUSQUE ROMANUS"

    def test_punctuation_preserved(self, normalizer):
        assert normalizer.normalize("Quid uis?") == "Quid vis?"
        assert normalizer.normalize("Veni, uidi, uici!") == "Veni, vidi, vici!"

    def test_numbers_preserved(self, normalizer):
        assert normalizer.normalize("anno 753 urbs condita") == \
               "anno 753 urbs condita"

    def test_single_character(self, normalizer):
        assert normalizer.normalize("u") == "u"

    def test_already_normalized(self, normalizer):
        text = "Arma virumque cano"
        assert normalizer.normalize(text) == text

    def test_macrons_handled(self, normalizer):
        assert normalizer.normalize("v\u012Bta") == "v\u012Bta"
        assert normalizer.normalize("u\u012Bta") == "v\u012Bta"
