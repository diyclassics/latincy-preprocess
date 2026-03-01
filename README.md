# latincy-preprocess

Latin text preprocessing: U/V normalization, long-s OCR correction, diacritics stripping, and macron removal — with optional Rust acceleration and spaCy integration.

Consolidates [latincy-uv](https://github.com/diyclassics/latincy-uv) and [latincy-long-s](https://github.com/diyclassics/latincy-long-s) into a single package.

## Installation

```bash
pip install latincy-preprocess
```

For spaCy pipeline components:
```bash
pip install latincy-preprocess[spacy]
```

## Quick Start

```python
from latincy_preprocess import normalize

normalize("Gallia eft omnis diuisa in partes tres")
# 'Gallia est omnis divisa in partes tres'
```

## Per-Normalizer Usage

### U/V Normalization

Converts u-only Latin spelling to proper u/v distinction using rule-based analysis:

```python
from latincy_preprocess import normalize_uv

normalize_uv("Arma uirumque cano")
# 'Arma virumque cano'
```

Rules handle digraphs (*qu*), trigraphs (*ngu*), morphological exceptions (*cui*, *fuit*), positional context (initial, intervocalic, post-consonant), and case preservation.

### Long-S OCR Correction

Corrects OCR errors where historical long-s (ſ) was misread as *f*, using n-gram frequency analysis from Latin treebank data:

```python
from latincy_preprocess import LongSNormalizer

normalizer = LongSNormalizer()

word, rules = normalizer.normalize_word_full("ftatua")
# ('statua', [TransformationRule(...)])

text = normalizer.normalize_text_full("funt in fundamento reipublicae ftatua")
# 'sunt in fundamento reipublicae statua'
```

Two-pass strategy: Pass 1 applies high-confidence rules (impossible bigrams like *ft*, *fp*, *fc*). Pass 2 uses 4-gram frequency disambiguation for ambiguous word-initial *f-* patterns.

### Diacritics and Macrons

```python
from latincy_preprocess import strip_diacritics, strip_macrons

strip_macrons("ārma")
# 'arma'

strip_diacritics("λόγος")
# 'λογος'
```

## spaCy Integration

Three pipeline components are available as spaCy factories:

### Unified Preprocessor (recommended)

Chains long-s correction → U/V normalization in the correct order:

```python
import spacy

nlp = spacy.blank("la")
nlp.add_pipe("latin_preprocessor")

doc = nlp("Gallia eft omnis diuisa in partes tres")
doc._.preprocessed          # 'Gallia est omnis divisa in partes tres'
doc[2]._.preprocessed       # 'est'
doc[2]._.preprocessed_lemma # normalized lemma
```

Either normalizer can be disabled:

```python
nlp.add_pipe("latin_preprocessor", config={"uv": False})
nlp.add_pipe("latin_preprocessor", config={"long_s": False})
```

### Standalone Components

```python
nlp.add_pipe("uv_normalizer")
# doc._.uv_normalized, token._.uv_normalized, token._.uv_normalized_lemma

nlp.add_pipe("long_s_normalizer")
# doc._.long_s_normalized, token._.long_s_normalized
```

## Rust Backend

When compiled with maturin, a Rust backend provides ~3x throughput for both normalizers. The backend is selected automatically:

```python
from latincy_preprocess import backend

backend()  # 'rust' or 'python'
```

The Python backend is fully functional and used as the fallback.

## Accuracy

### U/V Normalization

| Dataset | Accuracy |
|---------|----------|
| Curated test set (100 sentences) | 100% |
| UD Latin PROIEL (~21K u/v chars) | ~98% |
| UD Latin Perseus (~18K u/v chars) | ~97% |

### Long-S Correction

Pass 1 rules have a 0.00% false positive rate. Pass 2 disambiguation uses a protected allowlist of ~170 common Latin *f-* words (inline in `long_s/_rules.py`) plus n-gram frequency tables (JSON files in `long_s/data/ngrams/`).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Citation

```bibtex
@software{latincy_preprocess,
  title = {latincy-preprocess: Text Preprocessing for LatinCy Projects},
  author = {Burns, Patrick J.},
  year = {2026},
  url = {https://github.com/diyclassics/latincy-preprocess}
}
```

## License

MIT
