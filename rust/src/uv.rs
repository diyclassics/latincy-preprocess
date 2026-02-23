#[cfg(feature = "pyo3-backend")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3-backend")]
use pyo3::types::{PyDict, PyList};
use std::collections::HashSet;
use std::sync::LazyLock;

// =============================================================================
// Character Classification Helpers
// =============================================================================

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U'
            | '\u{0101}' // ā
            | '\u{0113}' // ē
            | '\u{012B}' // ī
            | '\u{014D}' // ō
            | '\u{016B}' // ū
            | '\u{0100}' // Ā
            | '\u{0112}' // Ē
            | '\u{012A}' // Ī
            | '\u{014C}' // Ō
            | '\u{016A}' // Ū
    )
}

fn is_consonant(c: char) -> bool {
    matches!(
        c.to_ascii_lowercase(),
        'b' | 'c' | 'd' | 'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'm' | 'n' | 'p' | 'q' | 'r'
            | 's' | 't' | 'w' | 'x' | 'y' | 'z'
    )
}

fn is_alpha(c: char) -> bool {
    c.is_alphabetic()
}

fn is_u_perfect_consonant(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'f' | 't' | 'n' | 'b' | 'c' | 'm' | 's' | 'p' | 'x')
}

fn is_word_boundary(chars: &[char], idx: usize) -> bool {
    if idx == 0 {
        return true;
    }
    !is_alpha(chars[idx - 1])
}

fn is_word_end(chars: &[char], idx: usize) -> bool {
    if idx == chars.len() - 1 {
        return true;
    }
    !is_alpha(chars[idx + 1])
}

fn extract_word(chars: &[char], idx: usize) -> String {
    let mut start = idx;
    while start > 0 && is_alpha(chars[start - 1]) {
        start -= 1;
    }
    let mut end = idx;
    while end < chars.len() - 1 && is_alpha(chars[end + 1]) {
        end += 1;
    }
    chars[start..=end]
        .iter()
        .map(|c| c.to_lowercase().next().unwrap_or(*c))
        .collect()
}

fn get_context(chars: &[char], idx: usize, window: usize) -> String {
    let start = idx.saturating_sub(window);
    let end = (idx + window + 1).min(chars.len());
    let mut result = String::new();
    for &c in &chars[start..idx] {
        result.push(c);
    }
    result.push('[');
    result.push(chars[idx]);
    result.push(']');
    for &c in &chars[idx + 1..end] {
        result.push(c);
    }
    result
}

// =============================================================================
// Word Exception Lists
// =============================================================================

static VOCALIC_U_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Demonstrative/relative pronouns
        "cui", "cuius", "huic", "huius", "cuique", "cuiquam",
        // Possessive pronouns (suus, tuus)
        "sua", "suae", "suam", "suas", "suis", "suo", "suos", "suum", "suorum", "suarum",
        "tua", "tuae", "tuam", "tuas", "tuis", "tuo", "tuos", "tuum", "tuorum", "tuarum",
        "tuus", "suus",
        // Other pronouns
        "eius", "eiusdem",
        // Numerals (duo)
        "duo", "duae", "duos", "duas", "duobus", "duabus", "duorum", "duarum",
        // Words with -uus/-uum pattern (vocalic u)
        "perpetuum", "perpetua", "perpetuae", "perpetuo", "perpetuam",
        "annuum", "annua", "annuae", "annuo",
        "mutuus", "mutua", "mutuae", "mutuum", "mutuo",
        "continuus", "continua", "continuae", "continuum", "continuo",
        "vacuus", "vacua", "vacuae", "vacuum", "vacuo",
        "ambiguus", "ambigua", "ambiguae", "ambiguum", "ambiguo",
        "exiguus", "exigua", "exiguum", "exiguo",
        "assiduus", "assidua", "assiduum", "assiduo",
        // U-perfect verb forms
        "intremuit", "tremuit", "fremuit", "gemuit", "intremuitque",
        "expalluit", "palluit",
        // Desero-type verbs
        "deseruit", "inseruit", "conseruit",
        // Syncopated perfects
        "potuere", "fuere", "habuere", "tenuere", "docuere", "monuere",
        "placuere", "tacuere", "patuere", "latuere", "caruere", "obstipuere",
        "obruerat", "obruit",
        // Fruor family
        "frui", "fruor", "fruitur", "fruuntur",
        // Other specific forms
        "tenues", "tenuis", "impluit", "compluit",
        // Fluo family
        "fluunt", "effluunt", "affluunt", "confluunt", "influunt",
        "refluunt", "defluunt", "profluunt", "circumfluunt",
    ]
    .into_iter()
    .collect()
});

// Stems where 'u' before vowel is vocalic (not consonantal).
// Covers all declined/conjugated forms via substring matching in Rule 10.
const VOCALIC_U_STEMS: &[&str] = &[
    "suad",      // suadeo, persuadeo
    "suar",      // suarum
    "suav",      // suavis
    "statu",     // statua, statuae, ...
    "ardu",      // ardua, arduum, ...
    "fatu",      // fatua, fatuum, ...
    "residu",    // residua, residuum, ...
    "strenu",    // strenua, strenuus, ...
    "conspicu",  // conspicua, conspicuum, ...
    "individu",  // individua, individuum, ...
];

// =============================================================================
// Core Classification Logic
// =============================================================================

/// Classify a u/v character at position idx.
/// Returns (normalized_char_lowercase, rule_name).
fn classify_uv(chars: &[char], idx: usize) -> (char, &'static str) {
    let c = chars[idx].to_lowercase().next().unwrap();
    debug_assert!(c == 'u' || c == 'v');

    let len = chars.len();

    // Helper closures for safe access
    let prev = if idx > 0 { Some(chars[idx - 1]) } else { None };
    let prev2 = if idx > 1 { Some(chars[idx - 2]) } else { None };
    let prev3 = if idx > 2 { Some(chars[idx - 3]) } else { None };
    let next1 = if idx + 1 < len { Some(chars[idx + 1]) } else { None };
    let next2 = if idx + 2 < len { Some(chars[idx + 2]) } else { None };
    let next3 = if idx + 3 < len { Some(chars[idx + 3]) } else { None };
    let next4 = if idx + 4 < len { Some(chars[idx + 4]) } else { None };
    let next5 = if idx + 5 < len { Some(chars[idx + 5]) } else { None };

    let word = extract_word(chars, idx);

    // Rule 1: After 'q' → ALWAYS 'u'
    if let Some(p) = prev {
        if p.to_ascii_lowercase() == 'q' {
            return ('u', "after_q");
        }
    }

    // Rule 2: 'ngu' before vowel → 'u' (digraph pattern)
    if let Some(p) = prev {
        if p.to_ascii_lowercase() == 'g' {
            if let Some(n) = next1 {
                if is_vowel(n) {
                    if let Some(p2) = prev2 {
                        if p2.to_ascii_lowercase() == 'n' {
                            return ('u', "ngu_digraph");
                        }
                    }
                    return ('u', "gu_before_vowel");
                }
            }
        }
    }

    // Rule 3: Word exceptions (morphological)
    if VOCALIC_U_WORDS.contains(word.as_str()) {
        return ('u', "word_exception");
    }

    // Rule 4: Perfect tense patterns
    // Special case: volo/nolo/malo have u-perfect with 'l'
    if let (Some(n1), Some(p)) = (next1, prev) {
        if n1.to_ascii_lowercase() == 'i' && p.to_ascii_lowercase() == 'l' {
            if word.starts_with("vol")
                || word.starts_with("nol")
                || word.starts_with("mal")
                || word.starts_with("uol")
            {
                if let Some(n2) = next2 {
                    if n2.to_ascii_lowercase() == 't' {
                        let n3_end = next3.map_or(true, |c| !is_alpha(c));
                        if n3_end {
                            return ('u', "volo_perfect");
                        }
                    }
                }
            }
        }
    }

    // Syncopated perfect -uere (3pl: potuere, fuere)
    if let (Some(n1), Some(n2), Some(n3)) = (next1, next2, next3) {
        if n1.to_ascii_lowercase() == 'e'
            && n2.to_ascii_lowercase() == 'r'
            && n3.to_ascii_lowercase() == 'e'
        {
            let n4_end = next4.map_or(true, |c| !is_alpha(c));
            if n4_end {
                if let Some(p) = prev {
                    if is_u_perfect_consonant(p) {
                        return ('u', "perfect_uere");
                    }
                }
            }
        }
    }

    // Standard -ui, -uit patterns
    if let Some(n1) = next1 {
        if n1.to_ascii_lowercase() == 'i' {
            // -ui at word end (1sg perfect: fui, potui)
            let n2_end = next2.map_or(true, |c| !is_alpha(c));
            if n2_end {
                if let Some(p) = prev {
                    if is_u_perfect_consonant(p) {
                        return ('u', "perfect_ui");
                    }
                }
            }

            // -uit at word end (3sg perfect: fuit, potuit)
            if let Some(n2) = next2 {
                if n2.to_ascii_lowercase() == 't' {
                    let n3_end = next3.map_or(true, |c| !is_alpha(c));
                    if n3_end {
                        if let Some(p) = prev {
                            if is_u_perfect_consonant(p) {
                                return ('u', "perfect_uit");
                            }
                        }
                    }
                }
            }

            // -uimus pattern (1pl perfect)
            if let (Some(n2), Some(n3), Some(n4)) = (next2, next3, next4) {
                if n2.to_ascii_lowercase() == 'm'
                    && n3.to_ascii_lowercase() == 'u'
                    && n4.to_ascii_lowercase() == 's'
                {
                    let n5_end = next5.map_or(true, |c| !is_alpha(c));
                    if n5_end {
                        if let Some(p) = prev {
                            if is_u_perfect_consonant(p) {
                                return ('u', "perfect_uimus");
                            }
                        }
                    }
                }
            }

            // Perfect -uisse (infinitive)
            if let (Some(n2), Some(n3), Some(n4)) = (next2, next3, next4) {
                if n2.to_ascii_lowercase() == 's'
                    && n3.to_ascii_lowercase() == 's'
                    && n4.to_ascii_lowercase() == 'e'
                {
                    let n5_end = next5.map_or(true, |c| !is_alpha(c));
                    if n5_end {
                        if let Some(p) = prev {
                            if is_consonant(p) {
                                return ('u', "perfect_uisse");
                            }
                        }
                    }
                }
            }
        }
    }

    // Perfect -uera-, -ueri-, -uero- (pluperfect/future perfect)
    if let (Some(n1), Some(n2), Some(n3)) = (next1, next2, next3) {
        if n1.to_ascii_lowercase() == 'e'
            && n2.to_ascii_lowercase() == 'r'
            && matches!(n3.to_ascii_lowercase(), 'a' | 'i' | 'o')
        {
            if let Some(p) = prev {
                if is_u_perfect_consonant(p) {
                    return ('u', "perfect_uer_stem");
                }
            }
        }
    }

    // Rule 5: Double-u patterns
    // FIRST u in uu sequence
    if let Some(n1) = next1 {
        if matches!(n1.to_ascii_lowercase(), 'u' | 'v') {
            if let Some(p) = prev {
                if is_consonant(p) {
                    if let Some(p2) = prev2 {
                        if is_vowel(p2) {
                            return ('v', "double_u_first_VCuu");
                        } else {
                            return ('u', "double_u_first_CCuu");
                        }
                    } else {
                        return ('u', "double_u_first_CCuu");
                    }
                } else if is_vowel(p) {
                    if p.to_ascii_lowercase() == 'i' && is_word_boundary(chars, idx - 1) {
                        return ('u', "double_u_first_initial_i");
                    } else {
                        return ('v', "double_u_first_Vuu");
                    }
                }
            }
        }
    }

    // SECOND u in uu sequence
    if let Some(p) = prev {
        if matches!(p.to_ascii_lowercase(), 'u' | 'v') {
            if let Some(p2) = prev2 {
                if is_consonant(p2) {
                    if let Some(p3) = prev3 {
                        if is_vowel(p3) {
                            return ('u', "double_u_second_VCuu");
                        } else {
                            return ('v', "double_u_second_CCuu");
                        }
                    } else {
                        return ('v', "double_u_second_CCuu");
                    }
                } else if is_vowel(p2) {
                    if p2.to_ascii_lowercase() == 'i' && is_word_boundary(chars, idx - 2) {
                        return ('v', "double_u_second_initial_i");
                    } else {
                        return ('u', "double_u_second_Vuu");
                    }
                }
            }
        }
    }

    // Rule 6: Word-initial before vowel → 'v'
    if is_word_boundary(chars, idx) {
        if let Some(n1) = next1 {
            if is_vowel(n1) {
                return ('v', "initial_before_vowel");
            }
        }
        return ('u', "initial_before_consonant");
    }

    // Rule 7: Intervocalic → 'v'
    if let (Some(p), Some(n1)) = (prev, next1) {
        if is_vowel(p) && is_vowel(n1) {
            return ('v', "intervocalic");
        }
    }

    // Rule 8: Before consonant → 'u'
    if let Some(n1) = next1 {
        if is_consonant(n1) {
            return ('u', "before_consonant");
        }
    }

    // Rule 9: Word-final → 'u'
    if is_word_end(chars, idx) {
        return ('u', "word_final");
    }

    // Rule 10: After consonant before vowel → 'v' (with vocalic stem exception)
    if let (Some(p), Some(n1)) = (prev, next1) {
        if is_consonant(p) && is_vowel(n1) {
            let word_lower = word.to_lowercase();
            for stem in VOCALIC_U_STEMS {
                if word_lower.contains(stem) {
                    return ('u', "vocalic_u_stem");
                }
            }
            return ('v', "post_consonant_before_vowel");
        }
    }

    // Rule 11: After consonant before consonant → 'u'
    if let Some(p) = prev {
        if is_consonant(p) {
            let next_is_consonant_or_end =
                next1.map_or(true, |n| is_consonant(n) || !is_alpha(n));
            if next_is_consonant_or_end {
                return ('u', "post_consonant_before_consonant");
            }
        }
    }

    // Default: keep as 'u' (conservative)
    ('u', "default")
}

// =============================================================================
// Public Rust API
// =============================================================================

pub fn normalize(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());

    for (i, &ch) in chars.iter().enumerate() {
        if matches!(ch.to_ascii_lowercase(), 'u' | 'v') {
            let (normalized, _) = classify_uv(&chars, i);
            if ch.is_uppercase() {
                result.push(normalized.to_uppercase().next().unwrap());
            } else {
                result.push(normalized);
            }
        } else {
            result.push(ch);
        }
    }

    result
}

pub fn normalize_char(text: &str, idx: usize) -> (String, &'static str) {
    let chars: Vec<char> = text.chars().collect();
    let ch = chars[idx];
    let (normalized, rule) = classify_uv(&chars, idx);

    let result_char = if ch.is_uppercase() {
        normalized.to_uppercase().collect()
    } else {
        normalized.to_string()
    };

    (result_char, rule)
}

pub struct DetailedResult {
    pub original: String,
    pub normalized: String,
    pub changes: Vec<ChangeRecord>,
}

pub struct ChangeRecord {
    pub position: usize,
    pub original: String,
    pub normalized: String,
    pub rule: &'static str,
    pub context: String,
}

pub fn normalize_detailed(text: &str) -> DetailedResult {
    if text.is_empty() {
        return DetailedResult {
            original: String::new(),
            normalized: String::new(),
            changes: Vec::new(),
        };
    }

    let chars: Vec<char> = text.chars().collect();
    let mut result_chars = String::with_capacity(text.len());
    let mut changes = Vec::new();

    for (i, &ch) in chars.iter().enumerate() {
        if matches!(ch.to_ascii_lowercase(), 'u' | 'v') {
            let (norm_lower, rule) = classify_uv(&chars, i);
            let normalized = if ch.is_uppercase() {
                norm_lower.to_uppercase().next().unwrap()
            } else {
                norm_lower
            };

            result_chars.push(normalized);

            if normalized != ch {
                changes.push(ChangeRecord {
                    position: i,
                    original: ch.to_string(),
                    normalized: normalized.to_string(),
                    rule,
                    context: get_context(&chars, i, 3),
                });
            }
        } else {
            result_chars.push(ch);
        }
    }

    DetailedResult {
        original: text.to_string(),
        normalized: result_chars,
        changes,
    }
}

// =============================================================================
// PyO3 wrappers
// =============================================================================

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
pub fn normalize_uv(text: &str) -> String {
    normalize(text)
}

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
pub fn normalize_uv_char(text: &str, idx: usize) -> (String, String) {
    let (ch, rule) = normalize_char(text, idx);
    (ch, rule.to_string())
}

#[cfg(feature = "pyo3-backend")]
#[pyfunction]
pub fn normalize_uv_detailed(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    let result = normalize_detailed(text);

    let dict = PyDict::new(py);
    dict.set_item("original", &result.original)?;
    dict.set_item("normalized", &result.normalized)?;

    let changes = PyList::empty(py);
    for change in &result.changes {
        let change_dict = PyDict::new(py);
        change_dict.set_item("position", change.position)?;
        change_dict.set_item("original", &change.original)?;
        change_dict.set_item("normalized", &change.normalized)?;
        change_dict.set_item("rule", change.rule)?;
        change_dict.set_item("context", &change.context)?;
        changes.append(change_dict)?;
    }
    dict.set_item("changes", changes)?;

    Ok(dict.into())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_after_q() {
        assert_eq!(normalize("quod"), "quod");
        assert_eq!(normalize("aqua"), "aqua");
        assert_eq!(normalize("quinque"), "quinque");
    }

    #[test]
    fn test_ngu_digraph() {
        assert_eq!(normalize("lingua"), "lingua");
        assert_eq!(normalize("sanguis"), "sanguis");
        assert_eq!(normalize("pinguis"), "pinguis");
    }

    #[test]
    fn test_word_exceptions() {
        assert_eq!(normalize("cui"), "cui");
        assert_eq!(normalize("sua"), "sua");
        assert_eq!(normalize("perpetuum"), "perpetuum");
        assert_eq!(normalize("eius"), "eius");
    }

    #[test]
    fn test_perfect_tense() {
        assert_eq!(normalize("fuit"), "fuit");
        assert_eq!(normalize("potuit"), "potuit");
        assert_eq!(normalize("fuisse"), "fuisse");
        assert_eq!(normalize("fuerat"), "fuerat");
        assert_eq!(normalize("voluit"), "voluit");
    }

    #[test]
    fn test_double_u() {
        assert_eq!(normalize("seruus"), "servus");
        assert_eq!(normalize("fluuius"), "fluvius");
        assert_eq!(normalize("nouus"), "novus");
        assert_eq!(normalize("iuuat"), "iuvat");
        assert_eq!(normalize("paruus"), "parvus");
    }

    #[test]
    fn test_initial_before_vowel() {
        assert_eq!(normalize("uia"), "via");
        assert_eq!(normalize("uir"), "vir");
        assert_eq!(normalize("uox"), "vox");
        assert_eq!(normalize("uinum"), "vinum");
    }

    #[test]
    fn test_intervocalic() {
        assert_eq!(normalize("nouo"), "novo");
        assert_eq!(normalize("breuis"), "brevis");
        assert_eq!(normalize("auis"), "avis");
    }

    #[test]
    fn test_sentence() {
        assert_eq!(
            normalize("Arma uirumque cano"),
            "Arma virumque cano"
        );
    }

    #[test]
    fn test_case_preservation() {
        assert_eq!(
            normalize("SENATVS POPVLVSQVE ROMANVS"),
            "SENATUS POPULUSQUE ROMANUS"
        );
    }

    #[test]
    fn test_soluit_distinguished() {
        assert_eq!(normalize("soluit"), "solvit");
    }

    #[test]
    fn test_vocalic_u_stems() {
        assert_eq!(normalize("statua"), "statua");
        assert_eq!(normalize("statuae"), "statuae");
        assert_eq!(normalize("ardua"), "ardua");
        assert_eq!(normalize("arduo"), "arduo");
        assert_eq!(normalize("fatua"), "fatua");
        assert_eq!(normalize("residua"), "residua");
        assert_eq!(normalize("strenua"), "strenua");
        assert_eq!(normalize("conspicua"), "conspicua");
        assert_eq!(normalize("individua"), "individua");
    }
}
