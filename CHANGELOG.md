# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-02-24

### Fixed

- Word-initial C+u clusters (e.g. *puer* → *pver*) misclassified as consonantal v.

## [0.1.1] - 2026-02-01

### Fixed

- `strip_diacritics()` no longer lowercases text — now preserves original case. Lowercasing was an unintended side effect conflating two separate operations.

## [0.1.0] - 2026-01-26

### Added

- Initial release: U/V normalization, long-s OCR correction, diacritics stripping, macron removal, spaCy integration, optional Rust backend.

[0.1.2]: https://github.com/diyclassics/latincy-preprocess/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/diyclassics/latincy-preprocess/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/diyclassics/latincy-preprocess/releases/tag/v0.1.0
