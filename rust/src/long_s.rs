#[cfg(feature = "pyo3-backend")]
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
#[cfg(feature = "pyo3-backend")]
use std::path::PathBuf;
use std::sync::LazyLock;

/// N-gram frequency tables, loaded lazily on first use.
struct NgramData {
    bigrams: HashMap<String, u64>,
    trigrams: HashMap<String, u64>,
    fourgrams: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Ngram data loading: two paths depending on feature flags
// ---------------------------------------------------------------------------

/// When pyo3-backend is NOT active, embed ngram JSON at compile time so the
/// CLI binary is fully self-contained.
#[cfg(not(feature = "pyo3-backend"))]
static NGRAM_DATA: LazyLock<NgramData> = LazyLock::new(|| {
    let bigrams: HashMap<String, u64> =
        serde_json::from_str(include_str!("../../src/latincy_preprocess/long_s/data/ngrams/bigrams.json"))
            .expect("embedded bigrams.json is invalid");
    let trigrams: HashMap<String, u64> =
        serde_json::from_str(include_str!("../../src/latincy_preprocess/long_s/data/ngrams/trigrams.json"))
            .expect("embedded trigrams.json is invalid");
    let fourgrams: HashMap<String, u64> =
        serde_json::from_str(include_str!("../../src/latincy_preprocess/long_s/data/ngrams/4grams.json"))
            .expect("embedded 4grams.json is invalid");
    NgramData {
        bigrams,
        trigrams,
        fourgrams,
    }
});

/// When pyo3-backend IS active, load ngram files at runtime from the Python
/// package's data directory (existing behavior).
#[cfg(feature = "pyo3-backend")]
static NGRAM_DATA: LazyLock<NgramData> = LazyLock::new(|| {
    let dir = find_ngram_dir();
    NgramData {
        bigrams: load_ngram_file(&dir.join("bigrams.json")),
        trigrams: load_ngram_file(&dir.join("trigrams.json")),
        fourgrams: load_ngram_file(&dir.join("4grams.json")),
    }
});

#[cfg(feature = "pyo3-backend")]
fn find_ngram_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LATINCY_PREPROCESS_NGRAMS") {
        return PathBuf::from(dir);
    }

    Python::with_gil(|py| {
        let module = py.import("latincy_preprocess.long_s._rules").ok()?;
        let file_attr = module.getattr("__file__").ok()?;
        let file_str: String = file_attr.extract().ok()?;
        let module_dir = PathBuf::from(file_str).parent()?.to_path_buf();
        Some(module_dir.join("data").join("ngrams"))
    })
    .unwrap_or_else(|| PathBuf::from("src/latincy_preprocess/long_s/data/ngrams"))
}

#[cfg(feature = "pyo3-backend")]
fn load_ngram_file(path: &std::path::Path) -> HashMap<String, u64> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read ngram file {}: {}", path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse ngram file {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Allowlist
// ---------------------------------------------------------------------------

/// Legitimate f-words that must not be transformed by Pass 2.
static ALLOWLIST: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "facere", "facio", "facit", "faciunt", "feceram", "fecerant", "fecerat", "fecere",
        "fecerim", "fecerint", "fecerit", "fecerunt", "feci", "fecimus", "fecisse", "fecissem",
        "fecissent", "fecisset", "fecisti", "fecistis", "fecit", "fecunda", "fecundam", "fecundi",
        "fecundis", "fecunditas", "fecunditatem", "fecundus", "felice", "felicem", "felices", "felici",
        "felicibus", "felicis", "feliciter", "felicium", "felix", "femina", "feminae", "feminam",
        "feminarum", "feminas", "feminis", "fenestra", "fenestram", "fenestras", "fenestris", "feram",
        "ferebam", "ferebant", "ferebat", "ferebatur", "feremus", "ferendi", "ferendo", "ferendum",
        "ferens", "ferent", "ferentem", "ferentis", "feres", "feret", "ferimus", "fero",
        "ferocem", "feroces", "feroci", "ferocis", "ferociter", "ferox", "ferre", "ferrem",
        "ferrent", "ferret", "ferri", "ferro", "ferrum", "fers", "fert", "fertis",
        "fertur", "ferunt", "feruntur", "festa", "festi", "festis", "festo", "festum",
        "fiant", "fiat", "fide", "fidei", "fideles", "fidelibus", "fidelis", "fideliter",
        "fidelium", "fidem", "fides", "fiebant", "fiebat", "fierent", "fieret", "fieri",
        "figura", "figurae", "figuram", "figurarum", "figuras", "figuris", "filia", "filiae",
        "filiam", "filiarum", "filias", "filii", "filiis", "filio", "filiorum", "filios",
        "filium", "filius", "finem", "fines", "finibus", "finire", "finis", "finit",
        "finita", "finitum", "finitur", "finium", "fio", "firma", "firmam", "firmamenti",
        "firmamento", "firmamentum", "firmare", "firmat", "firmi", "firmiter", "firmum", "firmus",
        "fit", "fiunt", "forma", "formae", "formam", "formas", "fuerat", "fuerint",
        "fuerit", "fuerunt", "fugere", "fugerunt", "fugi", "fugiens", "fugio", "fugisse",
        "fugit", "fugiunt", "fuisse", "fuissem", "fuissent", "fuisset", "fuit", "fundamenta",
        "fundamenti", "fundamento", "fundamentum", "furor", "furore", "furorem", "furoris", "futura",
        "futuram", "futuri", "futuris", "futurum", "futurus",
    ]
    .into_iter()
    .collect()
});

// ---------------------------------------------------------------------------
// Core normalization logic (always available)
// ---------------------------------------------------------------------------

fn pass1(word: &str) -> String {
    // Detect case pattern before lowercasing
    let chars: Vec<char> = word.chars().collect();
    let is_upper = chars.len() > 1 && chars.iter().all(|c| !c.is_lowercase());
    let is_title = chars.first().map_or(false, |c| c.is_uppercase())
        && (chars.len() == 1 || !is_upper);

    let mut normalized = word.to_lowercase();

    let trigram_rules: &[(&str, &str)] = &[
        ("fqu", "squ"),
        ("fpe", "spe"),
        ("fuf", "sus"),
        ("fum", "sum"),
    ];

    for &(pattern, replacement) in trigram_rules {
        if normalized.contains(pattern) {
            normalized = normalized.replace(pattern, replacement);
        }
    }

    let bigram_rules: &[(&str, &str)] = &[
        ("fp", "sp"),
        ("ft", "st"),
        ("fc", "sc"),
    ];

    for &(pattern, replacement) in bigram_rules {
        if normalized.contains(pattern) {
            normalized = normalized.replace(pattern, replacement);
        }
    }

    if normalized.ends_with('f') {
        let len = normalized.len();
        normalized.replace_range(len - 1..len, "s");
    }

    // Restore original case pattern
    if is_upper {
        normalized = normalized.to_uppercase();
    } else if is_title {
        let mut result = String::with_capacity(normalized.len());
        for (i, c) in normalized.chars().enumerate() {
            if i == 0 {
                result.extend(c.to_uppercase());
            } else {
                result.push(c);
            }
        }
        normalized = result;
    }

    normalized
}

fn restore_case(normalized: &str, is_upper: bool, is_title: bool) -> String {
    if is_upper {
        normalized.to_uppercase()
    } else if is_title {
        let mut result = String::with_capacity(normalized.len());
        for (i, c) in normalized.chars().enumerate() {
            if i == 0 {
                result.extend(c.to_uppercase());
            } else {
                result.push(c);
            }
        }
        result
    } else {
        normalized.to_string()
    }
}

fn pass2(word: &str, threshold: f64) -> String {
    // Detect case pattern before lowercasing
    let word_chars: Vec<char> = word.chars().collect();
    let is_upper = word_chars.len() > 1 && word_chars.iter().all(|c| !c.is_lowercase());
    let is_title = word_chars.first().map_or(false, |c| c.is_uppercase())
        && (word_chars.len() == 1 || !is_upper);

    let normalized = word.to_lowercase();
    let data = &*NGRAM_DATA;

    if ALLOWLIST.contains(normalized.as_str()) {
        return restore_case(&normalized, is_upper, is_title);
    }

    let chars: Vec<char> = normalized.chars().collect();

    if chars.len() >= 2 && chars[0] == 'f' && chars[1] == 'u' {
        let fu_freq = data.trigrams.get("<fu").copied().unwrap_or(0) as f64;
        let su_freq = data.trigrams.get("<su").copied().unwrap_or(0) as f64;

        if su_freq > fu_freq * threshold && su_freq > 0.0 {
            let mut result = String::with_capacity(normalized.len());
            result.push('s');
            result.extend(chars[1..].iter());
            return restore_case(&result, is_upper, is_title);
        }
    } else if chars.len() >= 2 && chars[0] == 'f' && chars[1] == 'e' {
        let fe_freq = data.trigrams.get("<fe").copied().unwrap_or(0) as f64;
        let se_freq = data.trigrams.get("<se").copied().unwrap_or(0) as f64;

        if se_freq > fe_freq * threshold && se_freq > 0.0 {
            let mut result = String::with_capacity(normalized.len());
            result.push('s');
            result.extend(chars[1..].iter());
            return restore_case(&result, is_upper, is_title);
        }
    } else if chars.len() >= 3 && chars[0] == 'f' && chars[1] == 'i' {
        let fi_key = format!("<fi{}", chars[2]);
        let si_key = format!("<si{}", chars[2]);
        let fi_freq = data.fourgrams.get(&fi_key).copied().unwrap_or(0) as f64;
        let si_freq = data.fourgrams.get(&si_key).copied().unwrap_or(0) as f64;

        if si_freq > fi_freq * threshold && si_freq > 0.0 {
            let mut result = String::with_capacity(normalized.len());
            result.push('s');
            result.extend(chars[1..].iter());
            return restore_case(&result, is_upper, is_title);
        }
    }

    restore_case(&normalized, is_upper, is_title)
}

// ---------------------------------------------------------------------------
// Public Rust API
// ---------------------------------------------------------------------------

pub fn normalize_word(word: &str, apply_pass2: bool) -> String {
    let result = pass1(word);
    if apply_pass2 {
        pass2(&result, 2.0)
    } else {
        result
    }
}

pub fn normalize_text(text: &str, apply_pass2: bool) -> String {
    text.split_whitespace()
        .map(|word| normalize_word(word, apply_pass2))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
pub fn normalize_long_s_word_pass1(word: &str) -> String {
    pass1(word)
}

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
#[pyo3(signature = (word, threshold=2.0))]
pub fn normalize_long_s_word_pass2(word: &str, threshold: f64) -> String {
    pass2(word, threshold)
}

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
#[pyo3(signature = (word, apply_pass2=true))]
pub fn normalize_long_s_word_full(word: &str, apply_pass2: bool) -> String {
    normalize_word(word, apply_pass2)
}

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
#[pyo3(signature = (text, apply_pass2=true))]
pub fn normalize_long_s_text_full(text: &str, apply_pass2: bool) -> String {
    normalize_text(text, apply_pass2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass1_trigrams() {
        assert_eq!(pass1("ftatua"), "statua");
        assert_eq!(pass1("fpiritus"), "spiritus");
        assert_eq!(pass1("fufcepit"), "suscepit");
        assert_eq!(pass1("fumma"), "summa");
        assert_eq!(pass1("fquama"), "squama");
    }

    #[test]
    fn test_pass1_bigrams() {
        assert_eq!(pass1("fpecies"), "species");
        assert_eq!(pass1("ftella"), "stella");
        assert_eq!(pass1("fcientia"), "scientia");
    }

    #[test]
    fn test_pass1_word_final() {
        assert_eq!(pass1("ef"), "es");
        assert_eq!(pass1("reuf"), "reus");
    }

    #[test]
    fn test_pass1_case_preservation() {
        assert_eq!(pass1("FTATUA"), "STATUA");
        assert_eq!(pass1("Fpiritus"), "Spiritus");
        assert_eq!(pass1("ftatua"), "statua");
    }

    #[test]
    fn test_normalize_word_pass1_only() {
        assert_eq!(normalize_word("ftatua", false), "statua");
        assert_eq!(normalize_word("fpiritus", false), "spiritus");
    }

    #[test]
    fn test_normalize_word_with_pass2() {
        assert_eq!(normalize_word("funt", true), "sunt");
    }

    #[test]
    fn test_normalize_text() {
        assert_eq!(
            normalize_text("ftatua fpiritus funt", true),
            "statua spiritus sunt"
        );
    }

    #[test]
    fn test_normalize_text_case_preservation() {
        assert_eq!(
            normalize_text("Sic uita eft", true),
            "Sic uita est"
        );
    }

    #[test]
    fn test_allowlist_preserved() {
        assert_eq!(normalize_word("fuit", true), "fuit");
    }

    #[test]
    fn test_allowlist_case_preserved() {
        assert_eq!(normalize_word("Fuit", true), "Fuit");
        assert_eq!(normalize_word("FUIT", true), "FUIT");
    }
}
