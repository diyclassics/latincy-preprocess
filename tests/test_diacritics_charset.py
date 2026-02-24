"""
Tests for the diacritics charset module.

Covers stripping, base_char extraction, CharsetMap round-trips,
and build_charset completeness.
"""

import json

import pytest

from latincy_preprocess.diacritics._charset import (
    CharsetMap,
    base_char,
    build_charset,
    strip_diacritics,
)


# =============================================================================
# strip_diacritics Tests
# =============================================================================


class TestStripDiacritics:
    def test_smooth_breathing_acute(self):
        assert strip_diacritics("ἄνθρωπος") == "ανθρωπος"

    def test_rough_breathing(self):
        assert strip_diacritics("ὁ") == "ο"

    def test_circumflex(self):
        assert strip_diacritics("τοῦ") == "του"

    def test_iota_subscript(self):
        assert strip_diacritics("τῷ") == "τω"

    def test_rough_breathing_rho(self):
        assert strip_diacritics("ῥήτωρ") == "ρητωρ"

    def test_diaeresis(self):
        assert strip_diacritics("προϊέναι") == "προιεναι"

    def test_grave_accent(self):
        assert strip_diacritics("τὸ") == "το"

    def test_combined_breathing_accent(self):
        # smooth breathing + acute
        assert strip_diacritics("ἔργον") == "εργον"
        # rough breathing + circumflex
        assert strip_diacritics("εὗρον") == "ευρον"

    def test_iota_subscript_with_accent(self):
        assert strip_diacritics("ᾷ") == "α"

    def test_already_stripped(self):
        assert strip_diacritics("λογος") == "λογος"

    def test_uppercase_preserved(self):
        assert strip_diacritics("ΛΟΓΟΣ") == "ΛΟΓΟΣ"

    def test_mixed_case_preserved(self):
        assert strip_diacritics("Ἀθῆναι") == "Αθηναι"

    def test_empty_string(self):
        assert strip_diacritics("") == ""

    def test_whitespace_preserved(self):
        assert strip_diacritics("ὁ λόγος") == "ο λογος"

    def test_punctuation_preserved(self):
        assert strip_diacritics("τί ἐστιν;") == "τι εστιν;"

    def test_full_sentence(self):
        result = strip_diacritics("ἐν ἀρχῇ ἦν ὁ λόγος")
        assert result == "εν αρχη ην ο λογος"

    def test_final_sigma_preserved(self):
        result = strip_diacritics("λόγος")
        assert result == "λογος"
        assert result[-1] == "ς"

    def test_non_greek_passthrough(self):
        assert strip_diacritics("hello") == "hello"
        assert strip_diacritics("123") == "123"


# =============================================================================
# base_char Tests
# =============================================================================


class TestBaseChar:
    def test_alpha_with_breathing(self):
        assert base_char("ἀ") == "α"
        assert base_char("ἁ") == "α"

    def test_alpha_with_accent(self):
        assert base_char("ά") == "α"
        assert base_char("ὰ") == "α"
        assert base_char("ᾶ") == "α"

    def test_alpha_with_iota_subscript(self):
        assert base_char("ᾳ") == "α"

    def test_plain_char(self):
        assert base_char("α") == "α"
        assert base_char("β") == "β"

    def test_uppercase(self):
        assert base_char("Ἀ") == "α"

    def test_space(self):
        assert base_char(" ") == " "

    def test_rho_breathing(self):
        assert base_char("ῥ") == "ρ"


# =============================================================================
# CharsetMap Tests
# =============================================================================


class TestCharsetMap:
    @pytest.fixture
    def simple_charset(self):
        return CharsetMap(
            input_chars=["<pad>", "<unk>", "α", "ο", " "],
            output_chars=["<pad>", "<unk>", "α", "ά", "ὰ", "ᾶ", "ο", "ό", "ὸ", " "],
            base_to_variants={
                "α": ["α", "ά", "ὰ", "ᾶ"],
                "ο": ["ο", "ό", "ὸ"],
                " ": [" "],
            },
        )

    def test_input_size(self, simple_charset):
        assert simple_charset.input_size == 5

    def test_output_size(self, simple_charset):
        assert simple_charset.output_size == 10

    def test_encode_input(self, simple_charset):
        indices = simple_charset.encode_input("αο")
        assert indices == [2, 3]

    def test_encode_output(self, simple_charset):
        indices = simple_charset.encode_output("άὸ")
        assert indices == [3, 8]

    def test_decode_output(self, simple_charset):
        text = simple_charset.decode_output([3, 8])
        assert text == "άὸ"

    def test_encode_unknown_falls_back(self, simple_charset):
        # Unknown chars map to <unk> index
        indices = simple_charset.encode_input("αβ")
        assert indices[0] == 2  # α
        assert indices[1] == simple_charset.input_to_idx["<unk>"]  # β → <unk>

    def test_save_load_roundtrip(self, simple_charset, tmp_path):
        path = tmp_path / "charset.json"
        simple_charset.save(path)

        loaded = CharsetMap.load(path)
        assert loaded.input_chars == simple_charset.input_chars
        assert loaded.output_chars == simple_charset.output_chars
        assert loaded.base_to_variants == simple_charset.base_to_variants

    def test_save_creates_parent_dirs(self, simple_charset, tmp_path):
        path = tmp_path / "nested" / "dir" / "charset.json"
        simple_charset.save(path)
        assert path.exists()

    def test_repr(self, simple_charset):
        r = repr(simple_charset)
        assert "input=5" in r
        assert "output=10" in r
        assert "bases=3" in r


# =============================================================================
# build_charset Tests
# =============================================================================


class TestBuildCharset:
    def test_simple_corpus(self):
        texts = ["ὁ λόγος", "τοῦ θεοῦ"]
        charset = build_charset(texts)

        # Should have <pad> and <unk> as first two
        assert charset.input_chars[0] == "<pad>"
        assert charset.input_chars[1] == "<unk>"
        assert charset.output_chars[0] == "<pad>"
        assert charset.output_chars[1] == "<unk>"

    def test_all_base_chars_present(self):
        texts = ["ὁ λόγος"]
        charset = build_charset(texts)

        # Should have all unique base chars from the input
        stripped = strip_diacritics("ὁ λόγος")
        for ch in set(stripped):
            assert ch in charset.input_to_idx

    def test_variants_grouped(self):
        texts = ["ὁ ὃ ὅ"]  # three omicrons with different diacritics
        charset = build_charset(texts)

        assert "ο" in charset.base_to_variants
        variants = charset.base_to_variants["ο"]
        assert "ὁ" in variants
        assert "ὃ" in variants
        assert "ὅ" in variants

    def test_encode_decode_roundtrip(self):
        texts = ["ἐν ἀρχῇ ἦν ὁ λόγος"]
        charset = build_charset(texts)

        original = "ἐν ἀρχῇ ἦν ὁ λόγος"
        encoded = charset.encode_output(original)
        decoded = charset.decode_output(encoded)
        assert decoded == original

    def test_input_encode_matches_strip(self):
        texts = ["ὁ λόγος"]
        charset = build_charset(texts)

        polytonic = "ὁ λόγος"
        stripped = strip_diacritics(polytonic)
        encoded = charset.encode_input(stripped)

        # Each index should map back to the stripped char
        for idx, ch in zip(encoded, stripped):
            assert charset.idx_to_input[idx] == ch

    def test_empty_corpus(self):
        charset = build_charset([])
        assert charset.input_size == 2  # just <pad>, <unk>
        assert charset.output_size == 2
