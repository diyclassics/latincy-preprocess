"""
Long-S normalization using transformation-based learning.

Multi-pass transformation system:
- Pass 1: High-confidence rules (0-1% ratio) - automatic corrections
- Pass 2: Context-dependent rules - frequency-based disambiguation

Based on n-gram frequency analysis of 842K Latin words.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class TransformationRule:
    """A character-level transformation rule."""
    pattern: str
    replacement: str
    description: str
    confidence: str  # "high", "medium", "low"
    ratio: float  # Frequency ratio from corpus analysis


class LongSNormalizer:
    """
    Normalize OCR long-s (ſ → f) artifacts in Latin text.

    Uses transformation-based learning with n-gram frequency validation.
    """

    def __init__(self, ngram_dir: Path = None):
        """
        Initialize normalizer with n-gram frequency tables.

        Args:
            ngram_dir: Path to directory containing n-gram JSON files.
                      Defaults to bundled package data.
        """
        if ngram_dir is None:
            ngram_dir = Path(__file__).parent / 'data' / 'ngrams'

        self.ngram_dir = Path(ngram_dir)

        # Load n-gram frequency tables
        self.bigrams = self._load_ngrams('bigrams.json')
        self.trigrams = self._load_ngrams('trigrams.json')
        self.fourgrams = self._load_ngrams('4grams.json')

        # Define transformation rules
        self.pass1_rules = self._define_pass1_rules()
        self.pass2_rules = self._define_pass2_rules()

        # Statistics tracking
        self.stats = {
            'total_words': 0,
            'words_modified': 0,
            'transformations': {},
        }

    def _load_ngrams(self, filename: str) -> Dict[str, int]:
        """Load n-gram frequency table from JSON."""
        filepath = self.ngram_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"N-gram file not found: {filepath}\n"
                "Run scripts/build_ngram_tables.py first to generate frequency tables."
            )

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _define_pass1_rules(self) -> List[TransformationRule]:
        """
        Define Pass 1 transformation rules (high-confidence, automatic).

        These patterns have < 1% occurrence in clean Latin corpus.
        Order matters: Apply longer patterns first to avoid partial matches.
        """
        return [
            # Trigram rules first (longer patterns take precedence)
            TransformationRule('fqu', 'squ', 'fqu \u2192 squ (impossible)', 'high', 0.00),
            TransformationRule('fpe', 'spe', 'fpe \u2192 spe (impossible)', 'high', 0.00),
            TransformationRule('fuf', 'sus', 'fuf \u2192 sus (0.18% ratio)', 'high', 0.18),
            TransformationRule('fum', 'sum', 'fum \u2192 sum (0.64% ratio)', 'high', 0.64),

            # Bigram rules (0.00% ratio)
            TransformationRule('fp', 'sp', 'fp \u2192 sp (impossible in Latin)', 'high', 0.00),
            TransformationRule('ft', 'st', 'ft \u2192 st (impossible in Latin)', 'high', 0.00),
            TransformationRule('fc', 'sc', 'fc \u2192 sc (impossible in Latin)', 'high', 0.00),

            # Word-final 'f' (0.01% ratio - only 6 occurrences in 842K words)
            # This is handled specially to avoid matching 'f' in middle of words
        ]

    def _define_pass2_rules(self) -> List[TransformationRule]:
        """
        Define Pass 2 transformation rules (context-dependent).

        These patterns need frequency-based disambiguation.
        """
        return [
            TransformationRule('fu', 'su', 'fu vs su (12.87% ratio)', 'medium', 12.87),
            TransformationRule('fe', 'se', 'fe vs se (23.30% ratio)', 'medium', 23.30),
            TransformationRule('fi', 'si', 'fi vs si (27.94% ratio)', 'low', 27.94),
        ]

    def normalize_word_pass1(self, word: str) -> Tuple[str, List[str]]:
        """
        Apply Pass 1 high-confidence transformations to a single word.

        Args:
            word: Input word (may contain OCR errors)

        Returns:
            (normalized_word, list_of_applied_rules)
        """
        # Detect case pattern before lowercasing
        is_upper = len(word) > 1 and word.isupper()
        is_title = word[0:1].isupper() and (len(word) == 1 or not word.isupper())

        normalized = word.lower()
        applied_rules = []

        # Apply bigram and trigram rules
        for rule in self.pass1_rules:
            if rule.pattern in normalized:
                normalized = normalized.replace(rule.pattern, rule.replacement)
                applied_rules.append(rule.description)

                # Track statistics
                rule_key = f"{rule.pattern} \u2192 {rule.replacement}"
                self.stats['transformations'][rule_key] = \
                    self.stats['transformations'].get(rule_key, 0) + 1

        # Special case: Word-final 'f' → 's'
        # Only 6 words ending in 'f' in 842K corpus (0.01% ratio)
        if normalized.endswith('f'):
            normalized = normalized[:-1] + 's'
            applied_rules.append('word-final f \u2192 s (0.01% ratio)')
            self.stats['transformations']['f> \u2192 s>'] = \
                self.stats['transformations'].get('f> \u2192 s>', 0) + 1

        # Restore original case pattern
        if is_upper:
            normalized = normalized.upper()
        elif is_title:
            normalized = normalized[0].upper() + normalized[1:]

        return normalized, applied_rules

    def normalize_word_pass2(self, word: str, threshold: float = 2.0) -> Tuple[str, List[str]]:
        """
        Apply Pass 2 context-dependent transformations using n-gram frequency.

        Args:
            word: Input word (after Pass 1)
            threshold: Frequency ratio threshold for transformation
                      (default: 2.0 = replace if 's' form is 2x more common)

        Returns:
            (normalized_word, list_of_applied_rules)
        """
        # Detect case pattern before lowercasing
        is_upper = len(word) > 1 and word.isupper()
        is_title = word[0:1].isupper() and (len(word) == 1 or not word.isupper())

        normalized = word.lower()
        applied_rules = []

        # Allowlist: known legitimate Latin words starting with 'fu', 'fe', 'fi'
        # Skip Pass 2 for these to avoid false positives (alpha-sorted)
        legitimate_f_words = {
            'facere', 'facio', 'facit', 'faciunt', 'feceram', 'fecerant', 'fecerat', 'fecere',
            'fecerim', 'fecerint', 'fecerit', 'fecerunt', 'feci', 'fecimus', 'fecisse', 'fecissem',
            'fecissent', 'fecisset', 'fecisti', 'fecistis', 'fecit', 'fecunda', 'fecundam', 'fecundi',
            'fecundis', 'fecunditas', 'fecunditatem', 'fecundus', 'felice', 'felicem', 'felices', 'felici',
            'felicibus', 'felicis', 'feliciter', 'felicium', 'felix', 'femina', 'feminae', 'feminam',
            'feminarum', 'feminas', 'feminis', 'fenestra', 'fenestram', 'fenestras', 'fenestris', 'feram',
            'ferebam', 'ferebant', 'ferebat', 'ferebatur', 'feremus', 'ferendi', 'ferendo', 'ferendum',
            'ferens', 'ferent', 'ferentem', 'ferentis', 'feres', 'feret', 'ferimus', 'fero',
            'ferocem', 'feroces', 'feroci', 'ferocis', 'ferociter', 'ferox', 'ferre', 'ferrem',
            'ferrent', 'ferret', 'ferri', 'ferro', 'ferrum', 'fers', 'fert', 'fertis',
            'fertur', 'ferunt', 'feruntur', 'festa', 'festi', 'festis', 'festo', 'festum',
            'fiant', 'fiat', 'fide', 'fidei', 'fideles', 'fidelibus', 'fidelis', 'fideliter',
            'fidelium', 'fidem', 'fides', 'fiebant', 'fiebat', 'fierent', 'fieret', 'fieri',
            'figura', 'figurae', 'figuram', 'figurarum', 'figuras', 'figuris', 'filia', 'filiae',
            'filiam', 'filiarum', 'filias', 'filii', 'filiis', 'filio', 'filiorum', 'filios',
            'filium', 'filius', 'finem', 'fines', 'finibus', 'finire', 'finis', 'finit',
            'finita', 'finitum', 'finitur', 'finium', 'fio', 'firma', 'firmam', 'firmamenti',
            'firmamento', 'firmamentum', 'firmare', 'firmat', 'firmi', 'firmiter', 'firmum', 'firmus',
            'fit', 'fiunt', 'forma', 'formae', 'formam', 'formas', 'fuerat', 'fuerint',
            'fuerit', 'fuerunt', 'fugere', 'fugerunt', 'fugi', 'fugiens', 'fugio', 'fugisse',
            'fugit', 'fugiunt', 'fuisse', 'fuissem', 'fuissent', 'fuisset', 'fuit', 'fundamenta',
            'fundamenti', 'fundamento', 'fundamentum', 'furor', 'furore', 'furorem', 'furoris', 'futura',
            'futuram', 'futuri', 'futuris', 'futurum', 'futurus',
        }

        if normalized in legitimate_f_words:
            # Restore case even for allowlisted words
            if is_upper:
                normalized = normalized.upper()
            elif is_title:
                normalized = normalized[0].upper() + normalized[1:]
            return normalized, applied_rules

        # Word-initial 'fu' vs 'su' disambiguation (use trigrams for '<fu' pattern)
        if normalized.startswith('fu') and len(normalized) >= 2:
            fu_trigram = f"<{normalized[:2]}"  # '<fu'
            su_trigram = f"<s{normalized[1]}"  # '<su'

            fu_freq = self.trigrams.get(fu_trigram, 0)
            su_freq = self.trigrams.get(su_trigram, 0)

            # If 'su' is significantly more common, transform
            if su_freq > fu_freq * threshold and su_freq > 0:
                normalized = 's' + normalized[1:]
                applied_rules.append(f'<fu \u2192 <su (freq: {su_freq} vs {fu_freq})')
                self.stats['transformations']['<fu \u2192 <su (Pass 2)'] = \
                    self.stats['transformations'].get('<fu \u2192 <su (Pass 2)', 0) + 1

        # Word-initial 'fe' vs 'se' disambiguation (use trigrams for '<fe' pattern)
        elif normalized.startswith('fe') and len(normalized) >= 2:
            fe_trigram = f"<{normalized[:2]}"  # '<fe'
            se_trigram = f"<s{normalized[1]}"  # '<se'

            fe_freq = self.trigrams.get(fe_trigram, 0)
            se_freq = self.trigrams.get(se_trigram, 0)

            if se_freq > fe_freq * threshold and se_freq > 0:
                normalized = 's' + normalized[1:]
                applied_rules.append(f'<fe \u2192 <se (freq: {se_freq} vs {fe_freq})')
                self.stats['transformations']['<fe \u2192 <se (Pass 2)'] = \
                    self.stats['transformations'].get('<fe \u2192 <se (Pass 2)', 0) + 1

        # Word-initial 'fi' vs 'si' disambiguation (use quadgrams for '<fi{x}' pattern)
        # Trigram ratio (<fi=7115 vs <si=18787) is only 2.6:1, too noisy.
        # Quadgrams are much more discriminating (e.g. <fim=13 vs <sim=2149).
        elif normalized.startswith('fi') and len(normalized) >= 3:
            fi_quadgram = f"<fi{normalized[2]}"  # e.g. '<fim' for 'fimulacra'
            si_quadgram = f"<si{normalized[2]}"  # e.g. '<sim' for 'simulacra'

            fi_freq = self.fourgrams.get(fi_quadgram, 0)
            si_freq = self.fourgrams.get(si_quadgram, 0)

            if si_freq > fi_freq * threshold and si_freq > 0:
                normalized = 's' + normalized[1:]
                applied_rules.append(f'<fi \u2192 <si (4gram freq: {si_freq} vs {fi_freq})')
                self.stats['transformations']['<fi \u2192 <si (Pass 2)'] = \
                    self.stats['transformations'].get('<fi \u2192 <si (Pass 2)', 0) + 1

        # Restore original case pattern
        if is_upper:
            normalized = normalized.upper()
        elif is_title:
            normalized = normalized[0].upper() + normalized[1:]

        return normalized, applied_rules

    def normalize_word_full(self, word: str, apply_pass2: bool = True) -> Tuple[str, List[str]]:
        """
        Apply all transformation passes to a word.

        Args:
            word: Input word (may contain OCR errors)
            apply_pass2: Whether to apply Pass 2 (context-dependent rules)

        Returns:
            (normalized_word, list_of_applied_rules)
        """
        # Pass 1: High-confidence rules
        normalized, rules_p1 = self.normalize_word_pass1(word)

        # Pass 2: Context-dependent rules
        if apply_pass2:
            normalized, rules_p2 = self.normalize_word_pass2(normalized)
            all_rules = rules_p1 + rules_p2
        else:
            all_rules = rules_p1

        return normalized, all_rules

    def normalize_text_pass1(self, text: str, report: bool = False) -> str:
        """
        Apply Pass 1 transformations to entire text.

        Args:
            text: Input text (space-separated words)
            report: If True, print transformation statistics

        Returns:
            Normalized text
        """
        words = text.split()
        normalized_words = []

        for word in words:
            self.stats['total_words'] += 1

            normalized, rules = self.normalize_word_pass1(word)

            if rules:
                self.stats['words_modified'] += 1

            normalized_words.append(normalized)

        result = ' '.join(normalized_words)

        if report:
            self.print_statistics()

        return result

    def normalize_text_full(self, text: str, apply_pass2: bool = True, report: bool = False) -> str:
        """
        Apply all transformation passes to entire text.

        Args:
            text: Input text (space-separated words)
            apply_pass2: Whether to apply Pass 2 (context-dependent rules)
            report: If True, print transformation statistics

        Returns:
            Normalized text
        """
        words = text.split()
        normalized_words = []

        for word in words:
            self.stats['total_words'] += 1

            normalized, rules = self.normalize_word_full(word, apply_pass2=apply_pass2)

            if rules:
                self.stats['words_modified'] += 1

            normalized_words.append(normalized)

        result = ' '.join(normalized_words)

        if report:
            self.print_statistics()

        return result

    def print_statistics(self):
        """Print transformation statistics."""
        total = self.stats['total_words']
        modified = self.stats['words_modified']

        if total == 0:
            print("No words processed.")
            return

        pct = (modified / total) * 100

        print("\n" + "="*60)
        print("TRANSFORMATION STATISTICS")
        print("="*60)
        print(f"Total words processed: {total:,}")
        print(f"Words modified: {modified:,} ({pct:.2f}%)")
        print(f"\nTransformations applied:")

        for rule, count in sorted(self.stats['transformations'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {rule:20} : {count:>6,}x")

        print("="*60 + "\n")

    def reset_statistics(self):
        """Reset transformation statistics."""
        self.stats = {
            'total_words': 0,
            'words_modified': 0,
            'transformations': {},
        }
