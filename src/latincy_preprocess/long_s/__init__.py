"""
Long-s normalization submodule.

Re-exports the rule engine and applies Rust backend acceleration
when available.
"""

from latincy_preprocess.long_s._rules import LongSNormalizer, TransformationRule

__all__ = ["LongSNormalizer", "TransformationRule"]


# Pass 1 rule descriptions (must match the originals from _define_pass1_rules
# so that tests checking `any("ft" in r for r in rules)` etc. still pass).
_PASS1_TRIGRAM_RULES = [
    ("fqu", "squ", "fqu \u2192 squ (impossible)"),
    ("fpe", "spe", "fpe \u2192 spe (impossible)"),
    ("fuf", "sus", "fuf \u2192 sus (0.18% ratio)"),
    ("fum", "sum", "fum \u2192 sum (0.64% ratio)"),
]
_PASS1_BIGRAM_RULES = [
    ("fp", "sp", "fp \u2192 sp (impossible in Latin)"),
    ("ft", "st", "ft \u2192 st (impossible in Latin)"),
    ("fc", "sc", "fc \u2192 sc (impossible in Latin)"),
]


def _apply_rust_backend(rust_module):
    """Monkey-patch Rust implementations onto LongSNormalizer."""

    def _rust_word_pass1(self, word):
        """Apply Pass 1 using Rust backend, with Python-compatible stats/rules."""
        result = rust_module.normalize_long_s_word_pass1(word)
        lower = word.lower()
        applied_rules = []

        if result != lower:
            tracking = lower

            for pattern, replacement, description in _PASS1_TRIGRAM_RULES:
                if pattern in tracking:
                    tracking = tracking.replace(pattern, replacement)
                    applied_rules.append(description)
                    rule_key = f"{pattern} \u2192 {replacement}"
                    self.stats['transformations'][rule_key] = \
                        self.stats['transformations'].get(rule_key, 0) + 1

            for pattern, replacement, description in _PASS1_BIGRAM_RULES:
                if pattern in tracking:
                    tracking = tracking.replace(pattern, replacement)
                    applied_rules.append(description)
                    rule_key = f"{pattern} \u2192 {replacement}"
                    self.stats['transformations'][rule_key] = \
                        self.stats['transformations'].get(rule_key, 0) + 1

            # Word-final f -> s
            if tracking.endswith('f'):
                tracking = tracking[:-1] + 's'
                applied_rules.append('word-final f \u2192 s (0.01% ratio)')
                self.stats['transformations']['f> \u2192 s>'] = \
                    self.stats['transformations'].get('f> \u2192 s>', 0) + 1

        return result, applied_rules

    def _rust_word_pass2(self, word, threshold=2.0):
        """Apply Pass 2 using Rust backend, with Python-compatible stats/rules."""
        result = rust_module.normalize_long_s_word_pass2(word, threshold)
        lower = word.lower()
        applied_rules = []

        if result != lower:
            if lower.startswith('fu') and result.startswith('su'):
                applied_rules.append(f'<fu \u2192 <su (freq: rust-accelerated)')
                self.stats['transformations']['<fu \u2192 <su (Pass 2)'] = \
                    self.stats['transformations'].get('<fu \u2192 <su (Pass 2)', 0) + 1
            elif lower.startswith('fe') and result.startswith('se'):
                applied_rules.append(f'<fe \u2192 <se (freq: rust-accelerated)')
                self.stats['transformations']['<fe \u2192 <se (Pass 2)'] = \
                    self.stats['transformations'].get('<fe \u2192 <se (Pass 2)', 0) + 1
            elif lower.startswith('fi') and result.startswith('si'):
                applied_rules.append(f'<fi \u2192 <si (4gram freq: rust-accelerated)')
                self.stats['transformations']['<fi \u2192 <si (Pass 2)'] = \
                    self.stats['transformations'].get('<fi \u2192 <si (Pass 2)', 0) + 1

        return result, applied_rules

    LongSNormalizer.normalize_word_pass1 = _rust_word_pass1
    LongSNormalizer.normalize_word_pass2 = _rust_word_pass2
