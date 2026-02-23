"""Tests for stripping utilities (strip_macrons, strip_diacritics, normalize_vu)."""

import pytest

from latincy_preprocess import normalize_vu, strip_macrons, strip_diacritics


# =============================================================================
# normalize_vu (V→U collapse)
# =============================================================================


class TestNormalizeVU:
    def test_basic(self):
        assert normalize_vu("Arma virumque cano") == "Arma uirumque cano"

    def test_uppercase(self):
        assert normalize_vu("VIRTVS") == "UIRTUS"

    def test_preserves_non_v(self):
        assert normalize_vu("consul") == "consul"

    def test_empty(self):
        assert normalize_vu("") == ""

    def test_round_trip_identity_direction(self):
        """VU then UV should recover v's (not necessarily identical, but v's restored)."""
        from latincy_preprocess import normalize_uv

        text = "uirumque"
        assert normalize_uv(text) == "virumque"
        assert normalize_vu("virumque") == "uirumque"

    def test_mixed_case(self):
        assert normalize_vu("Veni, vidi, vici") == "Ueni, uidi, uici"

    def test_only_v(self):
        assert normalize_vu("v") == "u"
        assert normalize_vu("V") == "U"


# =============================================================================
# strip_macrons
# =============================================================================


class TestStripMacrons:
    def test_precomposed_lowercase(self):
        assert strip_macrons("laudāre") == "laudare"

    def test_precomposed_uppercase(self):
        assert strip_macrons("RŌMA") == "ROMA"

    def test_mixed_case(self):
        assert strip_macrons("Rōma") == "Roma"

    def test_all_macron_vowels(self):
        assert strip_macrons("āēīōūȳ") == "aeiouy"
        assert strip_macrons("ĀĒĪŌŪȲ") == "AEIOUY"

    def test_combining_macron(self):
        # a + combining macron (U+0304)
        assert strip_macrons("a\u0304") == "a"

    def test_breve_stripped(self):
        # a + combining breve (U+0306)
        assert strip_macrons("a\u0306") == "a"

    def test_no_macrons_passthrough(self):
        assert strip_macrons("consul") == "consul"
        assert strip_macrons("") == ""

    def test_preserves_other_diacritics(self):
        """Non-macron diacritics should pass through."""
        assert strip_macrons("café") == "café"

    def test_mixed_macron_and_plain(self):
        assert strip_macrons("fēmina") == "femina"
        assert strip_macrons("cōnsul") == "consul"

    def test_multiple_macrons(self):
        assert strip_macrons("Rōmānōrum") == "Romanorum"


# =============================================================================
# strip_diacritics (Greek — re-exported from diacritics module)
# =============================================================================


class TestStripDiacriticsTopLevel:
    """Verify strip_diacritics is accessible from the package top level."""

    def test_basic_breathing(self):
        assert strip_diacritics("ἄνθρωπος") == "ανθρωπος"

    def test_circumflex(self):
        assert strip_diacritics("τοῦ") == "του"

    def test_iota_subscript(self):
        assert strip_diacritics("ᾳ") == "α"

    def test_passthrough_latin(self):
        assert strip_diacritics("consul") == "consul"

    def test_empty(self):
        assert strip_diacritics("") == ""
