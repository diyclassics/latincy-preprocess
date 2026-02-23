"""
Standalone rule-based U/V normalizer for Latin.

This module implements U/V normalization using phonotactic and morphological
rules. It has no external dependencies and can be used independently or
wrapped by spaCy or other frameworks.

Convention Reference (from Wiktionary Latin Entry Guidelines):
    "Prefer V for consonantal form, but prefer U for the vowel form."
    - V: consonantal sound, first sound in syllable
    - U: vowel sound, and in QU, GU, SU combinations

Example:
    >>> from latincy_preprocess.uv import normalize_uv
    >>> normalize_uv("Arma uirumque cano")
    'Arma virumque cano'

    >>> from latincy_preprocess.uv import UVNormalizerRules
    >>> normalizer = UVNormalizerRules()
    >>> normalizer.normalize("Veni, uidi, uici")
    'Veni, vidi, vici'
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["UVNormalizerRules", "NormalizationResult", "Change", "normalize_uv", "normalize_vu"]

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Change:
    """Record of a single normalization change."""

    position: int
    original: str
    normalized: str
    rule: str
    context: str


@dataclass
class NormalizationResult:
    """Detailed result from normalization."""

    original: str
    normalized: str
    changes: list[Change] = field(default_factory=list)

    @property
    def accuracy_vs(self, reference: str) -> float:
        """Calculate accuracy against a reference string."""
        if len(self.normalized) != len(reference):
            return 0.0
        matches = sum(a == b for a, b in zip(self.normalized, reference))
        return matches / len(reference)


# =============================================================================
# Character Classification Helpers
# =============================================================================

# Latin vowels (including macrons for compatibility)
_VOWELS = frozenset("aeiouāēīōūAEIOUĀĒĪŌŪ")

# Consonants (excluding u/v which are what we're classifying)
_CONSONANTS = frozenset("bcdfghjklmnpqrstwxyzBCDFGHJKLMNPQRSTWXYZ")


def _is_vowel(char: str) -> bool:
    """Check if character is a Latin vowel."""
    return char in _VOWELS


def _is_consonant(char: str) -> bool:
    """Check if character is a Latin consonant (excluding u/v)."""
    return char in _CONSONANTS


def _is_alpha(char: str) -> bool:
    """Check if character is alphabetic."""
    return char.isalpha()


def _get_char(text: str, idx: int) -> Optional[str]:
    """Safely get character at index, or None if out of bounds."""
    if 0 <= idx < len(text):
        return text[idx]
    return None


def _is_word_boundary(text: str, idx: int) -> bool:
    """Check if position is at word start (index 0 or after non-alpha)."""
    if idx == 0:
        return True
    prev = _get_char(text, idx - 1)
    return prev is None or not _is_alpha(prev)


def _is_word_end(text: str, idx: int) -> bool:
    """Check if position is at word end (last char or before non-alpha)."""
    if idx == len(text) - 1:
        return True
    next_char = _get_char(text, idx + 1)
    return next_char is None or not _is_alpha(next_char)


def _extract_word(text: str, idx: int) -> str:
    """Extract the word containing position idx."""
    start = idx
    while start > 0 and _is_alpha(_get_char(text, start - 1)):
        start -= 1
    end = idx
    while end < len(text) - 1 and _is_alpha(_get_char(text, end + 1)):
        end += 1
    return text[start : end + 1].lower()


def _get_context(text: str, idx: int, window: int = 3) -> str:
    """Get context string around position for debugging."""
    start = max(0, idx - window)
    end = min(len(text), idx + window + 1)
    return text[start:idx] + "[" + text[idx] + "]" + text[idx + 1 : end]


# =============================================================================
# Word Exception Lists
# =============================================================================

# Words with vocalic u that might otherwise be misclassified
_VOCALIC_U_WORDS = frozenset(
    {
        # Demonstrative/relative pronouns
        "cui",
        "cuius",
        "huic",
        "huius",
        "cuique",
        "cuiquam",
        # Possessive pronouns (suus, tuus)
        "sua",
        "suae",
        "suam",
        "suas",
        "suis",
        "suo",
        "suos",
        "suum",
        "suorum",
        "suarum",
        "tua",
        "tuae",
        "tuam",
        "tuas",
        "tuis",
        "tuo",
        "tuos",
        "tuum",
        "tuorum",
        "tuarum",
        "tuus",
        "suus",
        # Other pronouns
        "eius",
        "eiusdem",
        # Numerals (duo)
        "duo",
        "duae",
        "duos",
        "duas",
        "duobus",
        "duabus",
        "duorum",
        "duarum",
        # Words with -uus/-uum pattern (vocalic u)
        "perpetuum",
        "perpetua",
        "perpetuae",
        "perpetuo",
        "perpetuam",
        "annuum",
        "annua",
        "annuae",
        "annuo",
        "mutuus",
        "mutua",
        "mutuae",
        "mutuum",
        "mutuo",
        "continuus",
        "continua",
        "continuae",
        "continuum",
        "continuo",
        "vacuus",
        "vacua",
        "vacuae",
        "vacuum",
        "vacuo",
        "ambiguus",
        "ambigua",
        "ambiguae",
        "ambiguum",
        "ambiguo",
        "exiguus",
        "exigua",
        "exiguum",
        "exiguo",
        "assiduus",
        "assidua",
        "assiduum",
        "assiduo",
        # U-perfect verb forms
        "intremuit",
        "tremuit",
        "fremuit",
        "gemuit",
        "intremuitque",
        "expalluit",
        "palluit",
        # Desero-type verbs (u-perfect with 'r' stem)
        "deseruit",
        "inseruit",
        "conseruit",
        # Syncopated perfects (-uere for -uerunt)
        "potuere",
        "fuere",
        "habuere",
        "tenuere",
        "docuere",
        "monuere",
        "placuere",
        "tacuere",
        "patuere",
        "latuere",
        "caruere",
        "obstipuere",
        "obruerat",
        "obruit",
        # Fruor family (deponent)
        "frui",
        "fruor",
        "fruitur",
        "fruuntur",
        # Other specific forms
        "tenues",
        "tenuis",
        "impluit",
        "compluit",
        # Fluo family (-uunt endings have vocalic u's)
        "fluunt",
        "effluunt",
        "affluunt",
        "confluunt",
        "influunt",
        "refluunt",
        "defluunt",
        "profluunt",
        "circumfluunt",
    }
)

# Stems where 'u' before vowel is vocalic (not consonantal)
# These override the post_consonant_before_vowel rule
# Stems where 'u' before vowel is vocalic (not consonantal).
# These override Rule 10 (post_consonant_before_vowel → v).
# The stem check only fires when the u is in a C-u-V position, so
# false positives are structurally impossible: if u precedes a consonant,
# Rule 8 fires first; if u is intervocalic, Rule 7 fires first.
# This approach covers all declined/conjugated forms without enumerating them.
_VOCALIC_U_STEMS = frozenset(
    {
        "suad",      # suadeo, persuadeo (per-SUA-deo)
        "suar",      # suarum (sua + gen.pl.)
        "suav",      # suavis (sweet)
        "statu",     # statua, statuae, statuas, ...
        "ardu",      # ardua, arduum, arduo, ...
        "fatu",      # fatua, fatuum, fatuus, ...
        "residu",    # residua, residuum, ...
        "strenu",    # strenua, strenuus, ...
        "conspicu",  # conspicua, conspicuum, ...
        "individu",  # individua, individuum, ...
    }
)

# Consonants that typically precede u-perfect -ui- endings
# f (fuit), t (potuit), n (tenuit), b (habuit), c (docuit), m, s (posuit), p, x
_U_PERFECT_CONSONANTS = frozenset("ftnbcmspx")


# =============================================================================
# Core Classification Logic
# =============================================================================


def _classify_uv(text: str, idx: int) -> tuple[str, str]:
    """
    Classify a u/v character at position idx.

    Args:
        text: The full text
        idx: Position of the u/v character

    Returns:
        Tuple of (normalized_char, rule_name)
        normalized_char is 'u' or 'v' (lowercase)
    """
    char = text[idx].lower()
    if char not in ("u", "v"):
        raise ValueError(f"Expected u/v at position {idx}, got '{char}'")

    # Get surrounding context
    prev = _get_char(text, idx - 1)
    prev2 = _get_char(text, idx - 2)
    prev3 = _get_char(text, idx - 3)
    next1 = _get_char(text, idx + 1)
    next2 = _get_char(text, idx + 2)
    next3 = _get_char(text, idx + 3)
    next4 = _get_char(text, idx + 4)
    next5 = _get_char(text, idx + 5)

    # Extract current word for exception checking
    word = _extract_word(text, idx)

    # =========================================================================
    # Rule 1: After 'q' → ALWAYS 'u'
    # The 'qu' digraph represents /kw/, u is always vocalic
    # =========================================================================
    if prev and prev.lower() == "q":
        return ("u", "after_q")

    # =========================================================================
    # Rule 2: 'ngu' before vowel → 'u' (digraph pattern)
    # Examples: lingua, sanguis, pinguis
    # =========================================================================
    if prev and prev.lower() == "g":
        if next1 and _is_vowel(next1):
            if prev2 and prev2.lower() == "n":
                return ("u", "ngu_digraph")
            # Other gu + vowel: default to 'u' (digraph more common)
            return ("u", "gu_before_vowel")

    # =========================================================================
    # Rule 3: Word exceptions (morphological)
    # =========================================================================
    if word in _VOCALIC_U_WORDS:
        return ("u", "word_exception")

    # =========================================================================
    # Rule 4: Perfect tense patterns
    # =========================================================================

    # Special case: volo/nolo/malo have u-perfect with 'l'
    if next1 and next1.lower() == "i":
        if prev and prev.lower() == "l":
            if word.startswith(("vol", "nol", "mal", "uol")):
                if next2 and next2.lower() == "t":
                    if next3 is None or not _is_alpha(next3):
                        return ("u", "volo_perfect")

    # Syncopated perfect -uere (3pl: potuere, fuere)
    if next1 and next1.lower() == "e":
        if next2 and next2.lower() == "r":
            if next3 and next3.lower() == "e":
                if next4 is None or not _is_alpha(next4):
                    if prev and prev.lower() in _U_PERFECT_CONSONANTS:
                        return ("u", "perfect_uere")

    # Standard -ui, -uit patterns
    if next1 and next1.lower() == "i":
        # -ui at word end (1sg perfect: fui, potui)
        if next2 is None or not _is_alpha(next2):
            if prev and prev.lower() in _U_PERFECT_CONSONANTS:
                return ("u", "perfect_ui")

        # -uit at word end (3sg perfect: fuit, potuit)
        if next2 and next2.lower() == "t":
            if next3 is None or not _is_alpha(next3):
                if prev and prev.lower() in _U_PERFECT_CONSONANTS:
                    return ("u", "perfect_uit")

        # -uimus pattern (1pl perfect)
        if next2 and next2.lower() == "m":
            if next3 and next3.lower() == "u":
                if next4 and next4.lower() == "s":
                    if next5 is None or not _is_alpha(next5):
                        if prev and prev.lower() in _U_PERFECT_CONSONANTS:
                            return ("u", "perfect_uimus")

    # Perfect -uisse (infinitive)
    if next1 and next1.lower() == "i":
        if next2 and next2.lower() == "s":
            if next3 and next3.lower() == "s":
                if next4 and next4.lower() == "e":
                    if next5 is None or not _is_alpha(next5):
                        if prev and _is_consonant(prev):
                            return ("u", "perfect_uisse")

    # Perfect -uera-, -ueri-, -uero- (pluperfect/future perfect)
    if next1 and next1.lower() == "e":
        if next2 and next2.lower() == "r":
            if next3 and next3.lower() in "aio":
                if prev and prev.lower() in _U_PERFECT_CONSONANTS:
                    return ("u", "perfect_uer_stem")

    # =========================================================================
    # Rule 5: Double-u patterns
    # =========================================================================

    # FIRST u in uu sequence
    if next1 and next1.lower() in ("u", "v"):
        if prev and _is_consonant(prev):
            if prev2 and _is_vowel(prev2):
                # V-C-[u]-u → first u consonantal (servus)
                return ("v", "double_u_first_VCuu")
            else:
                # C-C-[u]-u → first u vocalic (fluvius)
                return ("u", "double_u_first_CCuu")
        elif prev and _is_vowel(prev):
            # V-[u]-u patterns
            if prev.lower() == "i" and _is_word_boundary(text, idx - 1):
                # Word-initial i-uu → first u vocalic (iuvat)
                return ("u", "double_u_first_initial_i")
            else:
                # Other V-uu → first u consonantal (novus)
                return ("v", "double_u_first_Vuu")

    # SECOND u in uu sequence
    if prev and prev.lower() in ("u", "v"):
        if prev2 and _is_consonant(prev2):
            if prev3 and _is_vowel(prev3):
                # V-C-u-[u] → second u vocalic (servus)
                return ("u", "double_u_second_VCuu")
            else:
                # C-C-u-[u] → second u consonantal (fluvius)
                return ("v", "double_u_second_CCuu")
        elif prev2 and _is_vowel(prev2):
            if prev2.lower() == "i" and _is_word_boundary(text, idx - 2):
                # i-u-[u] → second u consonantal (iuvat)
                return ("v", "double_u_second_initial_i")
            else:
                # V-u-[u] → second u vocalic (novus)
                return ("u", "double_u_second_Vuu")

    # =========================================================================
    # Rule 6: Word-initial before vowel → 'v'
    # Examples: via, virtus, vinum, vox
    # =========================================================================
    if _is_word_boundary(text, idx):
        if next1 and _is_vowel(next1):
            return ("v", "initial_before_vowel")
        # Word-initial before consonant → 'u' (urbs, usus)
        return ("u", "initial_before_consonant")

    # =========================================================================
    # Rule 7: Intervocalic → 'v'
    # Examples: novus, brevis, avis, caveo
    # =========================================================================
    if prev and _is_vowel(prev):
        if next1 and _is_vowel(next1):
            return ("v", "intervocalic")

    # =========================================================================
    # Rule 8: Before consonant → 'u'
    # Examples: scriptum, Augustus, causa
    # =========================================================================
    if next1 and _is_consonant(next1):
        return ("u", "before_consonant")

    # =========================================================================
    # Rule 9: Word-final → 'u'
    # Examples: tu, cum, dum, sum
    # =========================================================================
    if _is_word_end(text, idx):
        return ("u", "word_final")

    # =========================================================================
    # Rule 10: After consonant before vowel → 'v'
    # Examples: silva, servo, solvo
    # EXCEPTION: suad-, suav- stems have vocalic u (persuade, suavis)
    # =========================================================================
    if prev and _is_consonant(prev):
        if next1 and _is_vowel(next1):
            # Check for vocalic u stems (suad-, suav-)
            # Extract the word and check if it contains a vocalic stem pattern
            word_lower = word.lower()
            for vocalic_stem in _VOCALIC_U_STEMS:
                if vocalic_stem in word_lower:
                    return ("u", "vocalic_u_stem")
            return ("v", "post_consonant_before_vowel")

    # =========================================================================
    # Rule 11: After consonant before consonant → 'u'
    # Examples: vultus, cultus
    # =========================================================================
    if prev and _is_consonant(prev):
        if next1 is None or _is_consonant(next1) or not _is_alpha(next1):
            return ("u", "post_consonant_before_consonant")

    # =========================================================================
    # Default: keep as 'u' (conservative)
    # =========================================================================
    return ("u", "default")


# =============================================================================
# Main Normalizer Class
# =============================================================================


class UVNormalizerRules:
    """
    Rule-based U/V normalizer for Latin.

    Applies phonotactic and morphological rules to normalize u-only
    Latin spelling to proper u/v distinction.

    Example:
        >>> normalizer = UVNormalizerRules()
        >>> normalizer.normalize("Arma uirumque cano")
        'Arma virumque cano'
    """

    def normalize(self, text: str) -> str:
        """
        Normalize U/V in Latin text.

        Args:
            text: Latin text (u-only or mixed spelling)

        Returns:
            Text with proper U/V distinction applied
        """
        if not text:
            return text

        result = []
        for i, char in enumerate(text):
            if char.lower() in ("u", "v"):
                normalized, _ = _classify_uv(text, i)
                # Preserve original case
                result.append(normalized.upper() if char.isupper() else normalized)
            else:
                result.append(char)

        return "".join(result)

    def normalize_char(self, text: str, idx: int) -> tuple[str, str]:
        """
        Classify a single u/v character at position idx.

        Args:
            text: The full text (needed for context)
            idx: Position of the u/v character

        Returns:
            Tuple of (normalized_char, rule_name)
            Character preserves original case.

        Example:
            >>> normalizer = UVNormalizerRules()
            >>> normalizer.normalize_char("uia", 0)
            ('v', 'initial_before_vowel')
        """
        char = text[idx]
        if char.lower() not in ("u", "v"):
            raise ValueError(f"Character at position {idx} is not u/v: '{char}'")

        normalized, rule = _classify_uv(text, idx)
        # Preserve case
        if char.isupper():
            normalized = normalized.upper()
        return (normalized, rule)

    def normalize_detailed(self, text: str) -> NormalizationResult:
        """
        Normalize with full details about each decision.

        Args:
            text: Latin text to normalize

        Returns:
            NormalizationResult with original, normalized, and list of changes

        Example:
            >>> normalizer = UVNormalizerRules()
            >>> result = normalizer.normalize_detailed("uia")
            >>> result.normalized
            'via'
            >>> result.changes[0].rule
            'initial_before_vowel'
        """
        if not text:
            return NormalizationResult(original=text, normalized=text, changes=[])

        result_chars = []
        changes = []

        for i, char in enumerate(text):
            if char.lower() in ("u", "v"):
                normalized, rule = _classify_uv(text, i)
                # Preserve case
                if char.isupper():
                    normalized = normalized.upper()

                result_chars.append(normalized)

                # Record change if different
                if normalized != char:
                    changes.append(
                        Change(
                            position=i,
                            original=char,
                            normalized=normalized,
                            rule=rule,
                            context=_get_context(text, i),
                        )
                    )
            else:
                result_chars.append(char)

        return NormalizationResult(
            original=text, normalized="".join(result_chars), changes=changes
        )


# =============================================================================
# Module-level Convenience Function
# =============================================================================

# Singleton instance for convenience function
_default_normalizer: Optional[UVNormalizerRules] = None


def normalize_uv(text: str) -> str:
    """
    Normalize U/V in Latin text.

    Convenience function that uses a shared normalizer instance.

    Args:
        text: Latin text (u-only or mixed spelling)

    Returns:
        Text with proper U/V distinction applied

    Example:
        >>> normalize_uv("Arma uirumque cano")
        'Arma virumque cano'
    """
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = UVNormalizerRules()
    return _default_normalizer.normalize(text)


def normalize_vu(text: str) -> str:
    """
    Collapse V/U distinction back to u-only spelling.

    This is the reverse of normalize_uv(): every 'v' becomes 'u',
    preserving case. Useful for preparing text for models that
    expect u-only Latin or for round-trip testing.

    Args:
        text: Latin text with v/u distinction

    Returns:
        Text with all v→u (V→U) collapsed

    Example:
        >>> normalize_vu("Arma virumque cano")
        'Arma uirumque cano'
    """
    return text.replace("v", "u").replace("V", "U")
