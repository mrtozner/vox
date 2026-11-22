//! Deterministic text normalization for TTS.
//!
//! Two layers:
//!
//! - [`clean`] removes what no backend should ever be asked to speak: control
//!   characters, zero-width characters, emoji, Markdown markers. It also maps
//!   typographic quotes and dashes to plain ones.
//! - [`expand`] spells out numbers, dates, times, currency, units and common
//!   abbreviations in English or Turkish.
//!
//! Backends that carry their own text front end (Piper uses espeak-ng) only
//! need [`NormalizeMode::CleanOnly`]. Kokoro's bundled G2P has no number
//! handling worth the name, so Kokoro uses [`NormalizeMode::Full`].
//!
//! Everything here is a pure function of its input: no locale lookups, no
//! clock, no randomness.

/// Language whose expansion rules [`expand`] applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormLang {
    English,
    Turkish,
    /// No expansion rules; [`normalize`] only cleans.
    Other,
}

impl NormLang {
    /// Map a language tag (`en`, `en-US`, `en_GB`, `tr`, `tr_TR`, ...) to a rule set.
    pub fn from_tag(tag: &str) -> Self {
        let primary = tag
            .split(['-', '_'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        match primary.as_str() {
            "en" => Self::English,
            "tr" => Self::Turkish,
            _ => Self::Other,
        }
    }
}

/// How much [`normalize`] rewrites the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeMode {
    /// [`clean`] only.
    CleanOnly,
    /// [`clean`], then [`expand`].
    Full,
}

/// Normalize `text` for a TTS backend.
pub fn normalize(text: &str, lang: NormLang, mode: NormalizeMode) -> String {
    let cleaned = clean(text);
    match mode {
        NormalizeMode::CleanOnly => cleaned,
        NormalizeMode::Full => expand(&cleaned, lang),
    }
}

// ---------------------------------------------------------------------------
// Cleaning
// ---------------------------------------------------------------------------

fn is_emoji(c: char) -> bool {
    matches!(
        c as u32,
        0x1F000..=0x1FAFF      // pictographs, emoticons, transport, supplemental symbols
            | 0x2600..=0x27BF  // misc symbols, dingbats
            | 0x2B00..=0x2BFF  // misc symbols and arrows
            | 0x2190..=0x21FF  // arrows
            | 0xFE00..=0xFE0F  // variation selectors
            | 0x20E3           // combining keycap
            | 0xE0020..=0xE007F // tag characters (flag sequences)
    )
}

fn is_invisible(c: char) -> bool {
    matches!(
        c as u32,
        0x00AD | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 | 0xFEFF
    )
}

fn is_unicode_space(c: char) -> bool {
    matches!(
        c as u32,
        0x00A0 | 0x2000..=0x200A | 0x202F | 0x205F | 0x3000
    )
}

fn ends_sentence(s: &str) -> bool {
    s.trim_end().chars().last().is_some_and(|c| {
        matches!(
            c,
            '.' | '!' | '?' | ':' | ';' | ',' | '…' | '。' | '！' | '？'
        )
    })
}

/// Replace Markdown links `[label](target)` with `label`.
fn strip_markdown_links(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '['
            && let Some(close) = chars[i + 1..].iter().position(|&c| c == ']')
        {
            let close = i + 1 + close;
            if chars.get(close + 1) == Some(&'(')
                && let Some(end) = chars[close + 2..].iter().position(|&c| c == ')')
            {
                out.extend(&chars[i + 1..close]);
                i = close + 2 + end + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Strip a leading Markdown block marker (heading, bullet, block quote).
///
/// Returns the remaining text and whether a marker was removed.
fn strip_block_marker(line: &str) -> (&str, bool) {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if hashes > 0 && trimmed[hashes..].starts_with(' ') {
        return (trimmed[hashes..].trim_start(), true);
    }
    for marker in ["- ", "* ", "+ ", "> "] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return (rest.trim_start(), true);
        }
    }
    (trimmed, false)
}

fn clean_line(line: &str) -> String {
    let line = strip_markdown_links(line);
    let mut out = String::with_capacity(line.len());
    for c in line.chars() {
        match c {
            '\t' => out.push(' '),
            c if is_unicode_space(c) => out.push(' '),
            c if c.is_control() || is_invisible(c) || is_emoji(c) || c == '\u{200D}' => {}
            // Markdown emphasis and code markers.
            '*' | '`' => {}
            '_' => out.push(' '),
            '‘' | '’' | 'ʼ' | '′' => out.push('\''),
            '“' | '”' | '„' | '″' => out.push('"'),
            '–' | '‒' | '‑' | '‐' | '−' => out.push('-'),
            '―' => out.push('—'),
            c => out.push(c),
        }
    }
    out
}

/// Remove characters and markup that should never reach a synthesizer.
///
/// Line breaks become spaces. A paragraph break, a heading or a list item
/// that lacks closing punctuation gets a full stop so the pause survives.
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_break = false;

    for raw_line in text.split('\n') {
        let (body, had_marker) = strip_block_marker(raw_line.trim_end_matches('\r'));
        let line = clean_line(body);
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            pending_break = true;
            continue;
        }
        if !out.is_empty() {
            if (pending_break || had_marker) && !ends_sentence(&out) {
                out.push('.');
            }
            out.push(' ');
        }
        out.push_str(&line);
        // A heading or list item is its own utterance.
        pending_break = had_marker;
    }
    out
}

// ---------------------------------------------------------------------------
// Number words
// ---------------------------------------------------------------------------

const EN_ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const EN_SCALES: [&str; 7] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
];

const TR_ONES: [&str; 10] = [
    "sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
];
const TR_TENS: [&str; 10] = [
    "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
];
const TR_SCALES: [&str; 7] = [
    "",
    "bin",
    "milyon",
    "milyar",
    "trilyon",
    "katrilyon",
    "kentilyon",
];

fn en_below_hundred(n: u64) -> String {
    if n < 20 {
        EN_ONES[n as usize].to_string()
    } else if n.is_multiple_of(10) {
        EN_TENS[(n / 10) as usize].to_string()
    } else {
        format!(
            "{}-{}",
            EN_TENS[(n / 10) as usize],
            EN_ONES[(n % 10) as usize]
        )
    }
}

fn en_below_thousand(n: u64) -> String {
    let (hundreds, rest) = (n / 100, n % 100);
    match (hundreds, rest) {
        (0, r) => en_below_hundred(r),
        (h, 0) => format!("{} hundred", EN_ONES[h as usize]),
        (h, r) => format!("{} hundred {}", EN_ONES[h as usize], en_below_hundred(r)),
    }
}

fn tr_below_thousand(n: u64) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let (hundreds, tens, ones) = (n / 100, (n / 10) % 10, n % 10);
    if hundreds > 0 {
        // "yüz", not "bir yüz".
        if hundreds > 1 {
            parts.push(TR_ONES[hundreds as usize]);
        }
        parts.push("yüz");
    }
    if tens > 0 {
        parts.push(TR_TENS[tens as usize]);
    }
    if ones > 0 {
        parts.push(TR_ONES[ones as usize]);
    }
    parts.join(" ")
}

/// Spell out a cardinal number.
///
/// [`NormLang::Other`] uses the English words.
pub fn cardinal(n: u64, lang: NormLang) -> String {
    let turkish = lang == NormLang::Turkish;
    if n == 0 {
        return if turkish { TR_ONES[0] } else { EN_ONES[0] }.to_string();
    }
    // Three-digit groups, least significant first.
    let mut groups = Vec::new();
    let mut rest = n;
    while rest > 0 {
        groups.push(rest % 1000);
        rest /= 1000;
    }
    let mut parts = Vec::new();
    for (scale, &group) in groups.iter().enumerate().rev() {
        if group == 0 {
            continue;
        }
        if turkish {
            // "bin", not "bir bin"; but "bir milyon".
            if !(scale == 1 && group == 1) {
                parts.push(tr_below_thousand(group));
            }
            if scale > 0 {
                parts.push(TR_SCALES[scale].to_string());
            }
        } else {
            parts.push(en_below_thousand(group));
            if scale > 0 {
                parts.push(EN_SCALES[scale].to_string());
            }
        }
    }
    parts.join(" ")
}

/// Read a digit string one digit at a time ("007" -> "zero zero seven").
fn digit_by_digit(digits: &str, lang: NormLang) -> String {
    digits
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| cardinal(u64::from(d), lang))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Spell out an unsigned integer given as digits.
///
/// Leading zeros and values too large for `u64` are read digit by digit.
fn integer_words(digits: &str, lang: NormLang) -> String {
    if digits.len() > 1 && digits.starts_with('0') {
        return digit_by_digit(digits, lang);
    }
    match digits.parse::<u64>() {
        Ok(n) => cardinal(n, lang),
        Err(_) => digit_by_digit(digits, lang),
    }
}

/// English ordinal ("twenty-first") from a cardinal value.
fn en_ordinal(n: u64) -> String {
    let words = cardinal(n, NormLang::English);
    let split = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(split);
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{last}")
}

/// English reading of a calendar year ("twenty twenty-six").
fn en_year(year: u64) -> String {
    let (hi, lo) = (year / 100, year % 100);
    if year.is_multiple_of(1000) || (2001..=2009).contains(&year) {
        cardinal(year, NormLang::English)
    } else if lo == 0 {
        format!("{} hundred", en_below_hundred(hi))
    } else if lo < 10 {
        format!("{} oh {}", en_below_hundred(hi), EN_ONES[lo as usize])
    } else {
        format!("{} {}", en_below_hundred(hi), en_below_hundred(lo))
    }
}

/// English reading of a decade ("nineteen nineties").
fn en_decade(year: u64) -> String {
    let (hi, lo) = (year / 100, year % 100);
    if year.is_multiple_of(1000) {
        format!("{}s", cardinal(year, NormLang::English))
    } else if lo == 0 {
        format!("{} hundreds", en_below_hundred(hi))
    } else if lo < 20 {
        format!("{} tens", en_below_hundred(hi))
    } else {
        let tens = EN_TENS[(lo / 10) as usize];
        format!("{} {}ies", en_below_hundred(hi), &tens[..tens.len() - 1])
    }
}

const EN_MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const TR_MONTHS: [&str; 12] = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim",
    "Kasım", "Aralık",
];

fn date_words(day: u64, month: u64, year: u64, lang: NormLang) -> String {
    let idx = (month - 1) as usize;
    if lang == NormLang::Turkish {
        format!(
            "{} {} {}",
            cardinal(day, lang),
            TR_MONTHS[idx],
            cardinal(year, lang)
        )
    } else {
        format!("{} {}, {}", EN_MONTHS[idx], en_ordinal(day), en_year(year))
    }
}

// ---------------------------------------------------------------------------
// Abbreviations
// ---------------------------------------------------------------------------

struct Abbrev {
    /// Abbreviation as written, including its dots.
    short: &'static str,
    long: &'static str,
    /// Match regardless of case.
    any_case: bool,
    /// May close a sentence; when it does, the full stop is kept.
    terminal: bool,
}

const fn abbr(short: &'static str, long: &'static str, any_case: bool, terminal: bool) -> Abbrev {
    Abbrev {
        short,
        long,
        any_case,
        terminal,
    }
}

// Longer entries first where one is a prefix of another.
const EN_ABBREVS: &[Abbrev] = &[
    abbr("approx.", "approximately", true, false),
    abbr("Prof.", "Professor", false, false),
    abbr("Mrs.", "Missus", false, false),
    abbr("Mr.", "Mister", false, false),
    abbr("Ms.", "Miz", false, false),
    abbr("Dr.", "Doctor", false, false),
    abbr("Jr.", "Junior", false, true),
    abbr("Sr.", "Senior", false, true),
    abbr("e.g.", "for example", true, false),
    abbr("i.e.", "that is", true, false),
    abbr("etc.", "et cetera", true, true),
    abbr("vs.", "versus", true, false),
    abbr("a.m.", "AM", true, true),
    abbr("p.m.", "PM", true, true),
];

const TR_ABBREVS: &[Abbrev] = &[
    abbr("Prof.", "Profesör", false, false),
    abbr("Doç.", "Doçent", false, false),
    abbr("Dr.", "Doktor", false, false),
    abbr("Av.", "Avukat", false, false),
    abbr("Sn.", "Sayın", false, false),
    abbr("Cad.", "Caddesi", false, true),
    abbr("Sok.", "Sokak", false, true),
    abbr("bkz.", "bakınız", true, false),
    abbr("örn.", "örneğin", true, false),
    abbr("vb.", "ve benzeri", true, true),
    abbr("vs.", "vesaire", true, true),
    abbr("yy.", "yüzyıl", true, true),
];

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

struct Unit {
    symbol: &'static str,
    en_one: &'static str,
    en_many: &'static str,
    tr: &'static str,
}

// Longer symbols first.
const UNITS: &[Unit] = &[
    Unit {
        symbol: "km/h",
        en_one: "kilometer per hour",
        en_many: "kilometers per hour",
        tr: "kilometre bölü saat",
    },
    Unit {
        symbol: "mph",
        en_one: "mile per hour",
        en_many: "miles per hour",
        tr: "mil bölü saat",
    },
    Unit {
        symbol: "km",
        en_one: "kilometer",
        en_many: "kilometers",
        tr: "kilometre",
    },
    Unit {
        symbol: "kg",
        en_one: "kilogram",
        en_many: "kilograms",
        tr: "kilogram",
    },
    Unit {
        symbol: "cm",
        en_one: "centimeter",
        en_many: "centimeters",
        tr: "santimetre",
    },
    Unit {
        symbol: "mm",
        en_one: "millimeter",
        en_many: "millimeters",
        tr: "milimetre",
    },
    Unit {
        symbol: "ml",
        en_one: "milliliter",
        en_many: "milliliters",
        tr: "mililitre",
    },
];

struct Currency {
    symbol: char,
    en_one: &'static str,
    en_many: &'static str,
    en_cent_one: &'static str,
    en_cent_many: &'static str,
    tr: &'static str,
    tr_cent: &'static str,
}

const CURRENCIES: &[Currency] = &[
    Currency {
        symbol: '$',
        en_one: "dollar",
        en_many: "dollars",
        en_cent_one: "cent",
        en_cent_many: "cents",
        tr: "dolar",
        tr_cent: "sent",
    },
    Currency {
        symbol: '€',
        en_one: "euro",
        en_many: "euros",
        en_cent_one: "cent",
        en_cent_many: "cents",
        tr: "avro",
        tr_cent: "sent",
    },
    Currency {
        symbol: '£',
        en_one: "pound",
        en_many: "pounds",
        en_cent_one: "penny",
        en_cent_many: "pence",
        tr: "sterlin",
        tr_cent: "peni",
    },
    Currency {
        symbol: '₺',
        en_one: "lira",
        en_many: "liras",
        en_cent_one: "kurus",
        en_cent_many: "kurus",
        tr: "lira",
        tr_cent: "kuruş",
    },
];

fn currency_for(symbol: char) -> Option<&'static Currency> {
    CURRENCIES.iter().find(|c| c.symbol == symbol)
}

/// A number as written in the text.
struct Number {
    /// Integer digits with grouping separators removed.
    int: String,
    /// Fraction digits, if a decimal separator was present.
    frac: Option<String>,
    /// The decimal separator was a comma.
    comma_decimal: bool,
    /// Grouping separators were present.
    grouped: bool,
    /// Characters consumed.
    len: usize,
}

impl Number {
    fn is_plain_integer(&self) -> bool {
        self.frac.is_none() && !self.grouped
    }

    fn is_one(&self) -> bool {
        self.frac.is_none() && self.int.trim_start_matches('0') == "1"
    }

    fn words(&self, lang: NormLang) -> String {
        let int = integer_words(&self.int, lang);
        let Some(frac) = &self.frac else {
            return int;
        };
        if lang == NormLang::Turkish {
            let sep = if self.comma_decimal {
                "virgül"
            } else {
                "nokta"
            };
            // Leading zeros are read one by one, the rest as a number:
            // 3,14 -> "üç virgül on dört", 0,05 -> "sıfır virgül sıfır beş".
            let zeros = frac.len() - frac.trim_start_matches('0').len();
            let mut parts = vec![TR_ONES[0].to_string(); zeros];
            let rest = &frac[zeros..];
            if !rest.is_empty() {
                parts.push(integer_words(rest, lang));
            }
            format!("{int} {sep} {}", parts.join(" "))
        } else {
            format!("{int} point {}", digit_by_digit(frac, lang))
        }
    }
}

struct Expander<'a> {
    chars: &'a [char],
    lang: NormLang,
    out: String,
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric()
}

impl Expander<'_> {
    fn at(&self, i: usize) -> Option<char> {
        self.chars.get(i).copied()
    }

    fn digit_at(&self, i: usize) -> bool {
        self.at(i).is_some_and(|c| c.is_ascii_digit())
    }

    /// Number of consecutive ASCII digits starting at `i`.
    fn digit_run(&self, i: usize) -> usize {
        self.chars[i.min(self.chars.len())..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count()
    }

    fn digits(&self, i: usize, len: usize) -> String {
        self.chars[i..i + len].iter().collect()
    }

    fn value(&self, i: usize, len: usize) -> u64 {
        self.digits(i, len).parse().unwrap_or(u64::MAX)
    }

    /// True when position `i` is not inside a word.
    fn boundary_before(&self, i: usize) -> bool {
        i == 0 || !is_word_char(self.chars[i - 1])
    }

    fn boundary_after(&self, i: usize) -> bool {
        self.at(i).is_none_or(|c| !is_word_char(c))
    }

    /// Case-sensitive or case-insensitive literal match at `i`.
    fn matches(&self, i: usize, literal: &str, any_case: bool) -> Option<usize> {
        let mut n = 0;
        for want in literal.chars() {
            let got = self.at(i + n)?;
            let same = if any_case {
                got.to_lowercase().eq(want.to_lowercase())
            } else {
                got == want
            };
            if !same {
                return None;
            }
            n += 1;
        }
        Some(n)
    }

    fn push_word(&mut self, word: &str) {
        if self
            .out
            .chars()
            .last()
            .is_some_and(|c| !c.is_whitespace() && !matches!(c, '(' | '"' | '\'' | '-' | '/'))
        {
            self.out.push(' ');
        }
        self.out.push_str(word);
    }

    /// Like [`push_word`], but also separates from a following word.
    fn push_spaced(&mut self, word: &str, next: usize) {
        self.push_word(word);
        if self.at(next).is_some_and(|c| !c.is_whitespace()) {
            self.out.push(' ');
        }
    }

    /// The text continues with a new sentence at `i` (or ends there).
    fn sentence_break_at(&self, i: usize) -> bool {
        match self.at(i) {
            None => true,
            Some(c) if c.is_whitespace() => {
                let next = self.chars[i..].iter().find(|c| !c.is_whitespace());
                next.is_none_or(|c| c.is_uppercase())
            }
            _ => false,
        }
    }

    fn try_abbreviation(&mut self, i: usize) -> Option<usize> {
        if !self.boundary_before(i) {
            return None;
        }
        let table = match self.lang {
            NormLang::English => EN_ABBREVS,
            NormLang::Turkish => TR_ABBREVS,
            NormLang::Other => return None,
        };
        for entry in table {
            let Some(len) = self.matches(i, entry.short, entry.any_case) else {
                continue;
            };
            if !self.boundary_after(i + len) {
                continue;
            }
            self.push_word(entry.long);
            if entry.terminal && self.sentence_break_at(i + len) {
                self.out.push('.');
            }
            return Some(len);
        }

        // "St." is Saint before a capitalized name, Street otherwise. A sentence
        // that ends in "St." followed by another one is misread as Saint.
        if self.lang == NormLang::English
            && let Some(len) = self.matches(i, "St.", false)
            && self.boundary_after(i + len)
        {
            let before_name = self.at(i + len).is_some_and(char::is_whitespace)
                && self.at(i + len + 1).is_some_and(char::is_uppercase);
            if before_name {
                self.push_word("Saint");
            } else {
                self.push_word("Street");
                if self.at(i + len).is_none() {
                    self.out.push('.');
                }
            }
            return Some(len);
        }

        // "No. 5" / "No.5": only before a digit.
        if let Some(len) = self.matches(i, "No.", false) {
            let mut j = i + len;
            while self.at(j) == Some(' ') {
                j += 1;
            }
            if self.digit_at(j) {
                self.push_word(if self.lang == NormLang::Turkish {
                    "Numara"
                } else {
                    "Number"
                });
                return Some(len);
            }
        }
        None
    }

    /// Parse a number starting at `i` (which must be a digit).
    fn parse_number(&self, i: usize) -> Number {
        let turkish = self.lang == NormLang::Turkish;
        let group_sep = if turkish { '.' } else { ',' };
        let first = self.digit_run(i);
        let mut int = self.digits(i, first);
        let mut j = i + first;
        let mut grouped = false;

        if (1..=3).contains(&first) && !int.starts_with('0') {
            while self.at(j) == Some(group_sep) && self.digit_run(j + 1) == 3 {
                int.push_str(&self.digits(j + 1, 3));
                j += 4;
                grouped = true;
            }
        }

        let mut frac = None;
        let mut comma_decimal = false;
        match self.at(j) {
            Some('.') if self.digit_at(j + 1) => {
                let n = self.digit_run(j + 1);
                frac = Some(self.digits(j + 1, n));
                j += 1 + n;
            }
            Some(',') if turkish && self.digit_at(j + 1) => {
                let n = self.digit_run(j + 1);
                frac = Some(self.digits(j + 1, n));
                comma_decimal = true;
                j += 1 + n;
            }
            _ => {}
        }

        Number {
            int,
            frac,
            comma_decimal,
            grouped,
            len: j - i,
        }
    }

    /// `YYYY-MM-DD`, `D.M.YYYY` or `D/M/YYYY` (`M/D/YYYY` in English).
    fn try_date(&mut self, i: usize) -> Option<usize> {
        let a = self.digit_run(i);
        let sep = self.at(i + a)?;
        let b = self.digit_run(i + a + 1);
        if self.at(i + a + 1 + b) != Some(sep) {
            return None;
        }
        let c = self.digit_run(i + a + b + 2);
        let len = a + b + c + 2;
        let (va, vb, vc) = (
            self.value(i, a),
            self.value(i + a + 1, b),
            self.value(i + a + b + 2, c),
        );
        let (day, month, year) = match sep {
            '-' if a == 4 && b == 2 && c == 2 => (vc, vb, va),
            '.' if (1..=2).contains(&a) && (1..=2).contains(&b) && c == 4 => (va, vb, vc),
            '/' if (1..=2).contains(&a) && (1..=2).contains(&b) && c == 4 => {
                if self.lang == NormLang::Turkish {
                    (va, vb, vc)
                } else {
                    (vb, va, vc)
                }
            }
            _ => return None,
        };
        if !(1..=31).contains(&day) || !(1..=12).contains(&month) || year == 0 {
            return None;
        }
        self.push_word(&date_words(day, month, year, self.lang));
        Some(len)
    }

    /// `H:MM` or `H:MM:SS`, optionally followed by am/pm.
    fn try_time(&mut self, i: usize) -> Option<usize> {
        let h = self.digit_run(i);
        if !(1..=2).contains(&h) || self.at(i + h) != Some(':') || self.digit_run(i + h + 1) != 2 {
            return None;
        }
        let hour = self.value(i, h);
        let minute = self.value(i + h + 1, 2);
        if hour > 23 || minute > 59 {
            return None;
        }
        let mut len = h + 3;
        let mut second = None;
        if self.at(i + len) == Some(':') && self.digit_run(i + len + 1) == 2 {
            let s = self.value(i + len + 1, 2);
            if s <= 59 {
                second = Some(s);
                len += 3;
            }
        }

        let lang = self.lang;
        let mut words = cardinal(hour, lang);
        if lang == NormLang::Turkish {
            if minute > 0 {
                if minute < 10 {
                    words.push_str(" sıfır");
                }
                words.push(' ');
                words.push_str(&cardinal(minute, lang));
            }
            if let Some(s) = second {
                words.push_str(&format!(" {} saniye", cardinal(s, lang)));
            }
        } else {
            // Optional meridiem: "3:30 pm", "3:30pm", "3:30 p.m."
            let mut j = i + len;
            if self.at(j) == Some(' ') {
                j += 1;
            }
            let meridiem = [("a.m.", "AM"), ("p.m.", "PM"), ("am", "AM"), ("pm", "PM")]
                .into_iter()
                .find_map(|(text, spoken)| {
                    let n = self.matches(j, text, true)?;
                    self.boundary_after(j + n).then_some((n, spoken, text))
                });

            if minute == 0 {
                if meridiem.is_none() {
                    words.push_str(" o'clock");
                }
            } else if minute < 10 {
                words.push_str(&format!(" oh {}", cardinal(minute, lang)));
            } else {
                words.push_str(&format!(" {}", cardinal(minute, lang)));
            }
            if let Some(s) = second {
                let unit = if s == 1 { "second" } else { "seconds" };
                words.push_str(&format!(" and {} {unit}", cardinal(s, lang)));
            }
            if let Some((n, spoken, text)) = meridiem {
                words.push(' ');
                words.push_str(spoken);
                let end = j + n;
                // "p.m." closing a sentence keeps the full stop.
                if text.ends_with('.') && self.sentence_break_at(end) {
                    words.push('.');
                }
                len = end - i;
            }
        }
        self.push_word(&words);
        Some(len)
    }

    fn money_words(&self, number: &Number, currency: &Currency) -> String {
        let lang = self.lang;
        let turkish = lang == NormLang::Turkish;
        let whole = integer_words(&number.int, lang);
        let whole_is_one = number.int.trim_start_matches('0') == "1";
        let major = |one: bool| {
            if turkish {
                currency.tr
            } else if one {
                currency.en_one
            } else {
                currency.en_many
            }
        };
        match &number.frac {
            None => format!("{whole} {}", major(whole_is_one)),
            Some(frac) if frac.len() == 2 => {
                let cents: u64 = frac.parse().unwrap_or(0);
                if cents == 0 {
                    return format!("{whole} {}", major(whole_is_one));
                }
                let cent_words = cardinal(cents, lang);
                if turkish {
                    format!("{whole} {} {cent_words} {}", currency.tr, currency.tr_cent)
                } else {
                    let minor = if cents == 1 {
                        currency.en_cent_one
                    } else {
                        currency.en_cent_many
                    };
                    format!("{whole} {} and {cent_words} {minor}", major(whole_is_one))
                }
            }
            // "$3.5" and the like: read the number, then the unit.
            Some(_) => format!("{} {}", number.words(lang), major(false)),
        }
    }

    /// Expand the number at `i` together with whatever qualifies it.
    fn expand_number(&mut self, i: usize) -> usize {
        if let Some(len) = self.try_date(i) {
            return len;
        }
        if let Some(len) = self.try_time(i) {
            return len;
        }

        let lang = self.lang;
        let turkish = lang == NormLang::Turkish;
        let number = self.parse_number(i);
        let end = i + number.len;
        let next = self.at(end);

        // Percent: "25%".
        if next == Some('%') {
            let words = number.words(lang);
            self.push_word(&if turkish {
                format!("yüzde {words}")
            } else {
                format!("{words} percent")
            });
            return number.len + 1;
        }

        // Degrees: "3°", "21°C".
        if next == Some('°') {
            let mut len = number.len + 1;
            let scale = match self.at(end + 1) {
                Some('C') if self.boundary_after(end + 2) => Some(("Celsius", "santigrat")),
                Some('F') if self.boundary_after(end + 2) => Some(("Fahrenheit", "fahrenhayt")),
                _ => None,
            };
            let mut words = number.words(lang);
            if turkish {
                words.push_str(" derece");
            } else if number.is_one() {
                words.push_str(" degree");
            } else {
                words.push_str(" degrees");
            }
            if let Some((en, tr)) = scale {
                words.push(' ');
                words.push_str(if turkish { tr } else { en });
                len += 1;
            }
            self.push_word(&words);
            return len;
        }

        // Trailing currency: "5$", "5€", "5 TL".
        if let Some(currency) = next.and_then(currency_for)
            && self.boundary_after(end + 1)
        {
            let words = self.money_words(&number, currency);
            self.push_word(&words);
            return number.len + 1;
        }
        {
            let j = if next == Some(' ') { end + 1 } else { end };
            if let Some(n) = self.matches(j, "TL", false)
                && self.boundary_after(j + n)
            {
                let lira = currency_for('₺').expect("lira is in the currency table");
                let words = self.money_words(&number, lira);
                self.push_word(&words);
                return j + n - i;
            }
        }

        // Units: "5 km", "5km".
        {
            let j = if next == Some(' ') { end + 1 } else { end };
            for unit in UNITS {
                let Some(n) = self.matches(j, unit.symbol, false) else {
                    continue;
                };
                if !self.boundary_after(j + n) {
                    continue;
                }
                let name = if turkish {
                    unit.tr
                } else if number.is_one() {
                    unit.en_one
                } else {
                    unit.en_many
                };
                self.push_word(&format!("{} {name}", number.words(lang)));
                return j + n - i;
            }
        }

        if lang != NormLang::Turkish && number.is_plain_integer() {
            let value = number.int.parse::<u64>().ok();

            // Ordinals: "1st", "22nd".
            if let Some(v) = value {
                for suffix in ["st", "nd", "rd", "th"] {
                    if let Some(n) = self.matches(end, suffix, true)
                        && self.boundary_after(end + n)
                    {
                        self.push_word(&en_ordinal(v));
                        return number.len + n;
                    }
                }
            }

            // Years and decades: "2026", "1990s".
            if let Some(year) = value.filter(|v| number.int.len() == 4 && (1100..=2099).contains(v))
            {
                if self.at(end) == Some('s') && self.boundary_after(end + 1) && year % 10 == 0 {
                    self.push_word(&en_decade(year));
                    return number.len + 1;
                }
                self.push_word(&en_year(year));
                return number.len;
            }
        }

        self.push_word(&number.words(lang));

        // Turkish case suffix: "2026'da" -> "iki bin yirmi altıda".
        if turkish && next == Some('\'') && self.at(end + 1).is_some_and(char::is_alphabetic) {
            let suffix: String = self.chars[end + 1..]
                .iter()
                .take_while(|c| c.is_alphabetic())
                .collect();
            self.out.push_str(&suffix);
            return number.len + 1 + suffix.chars().count();
        }
        number.len
    }

    fn run(mut self) -> String {
        let lang = self.lang;
        let turkish = lang == NormLang::Turkish;
        let mut i = 0;
        while i < self.chars.len() {
            let c = self.chars[i];

            if c.is_alphabetic() {
                if let Some(len) = self.try_abbreviation(i) {
                    i += len;
                    continue;
                }
                // Copy the whole word so abbreviations only match at word starts.
                while i < self.chars.len() && self.chars[i].is_alphabetic() {
                    self.out.push(self.chars[i]);
                    i += 1;
                }
                continue;
            }

            if c.is_ascii_digit() {
                // "B12" -> "B twelve": a number glued to letters is still a number.
                i += self.expand_number(i);
                if self.at(i).is_some_and(char::is_alphanumeric) {
                    self.out.push(' ');
                }
                continue;
            }

            match c {
                // Leading currency: "$5", "€1.50".
                c if currency_for(c).is_some() && self.digit_at(i + 1) => {
                    let currency = currency_for(c).expect("checked above");
                    let number = self.parse_number(i + 1);
                    let words = self.money_words(&number, currency);
                    self.push_word(&words);
                    i += 1 + number.len;
                }
                // Leading percent: "%25".
                '%' if self.digit_at(i + 1) => {
                    let number = self.parse_number(i + 1);
                    let words = number.words(lang);
                    self.push_word(&if turkish {
                        format!("yüzde {words}")
                    } else {
                        format!("{words} percent")
                    });
                    i += 1 + number.len;
                }
                '%' => {
                    self.push_spaced(if turkish { "yüzde" } else { "percent" }, i + 1);
                    i += 1;
                }
                '#' if self.digit_at(i + 1) => {
                    self.push_spaced(if turkish { "numara" } else { "number" }, i + 1);
                    i += 1;
                }
                // Sign: "-5" at the start of a token.
                '-' if self.digit_at(i + 1)
                    && (i == 0
                        || self.chars[i - 1].is_whitespace()
                        || self.chars[i - 1] == '(') =>
                {
                    self.push_spaced(if turkish { "eksi" } else { "minus" }, i + 1);
                    i += 1;
                }
                '+' if self.digit_at(i + 1) || self.at(i + 1) == Some(' ') => {
                    self.push_spaced(if turkish { "artı" } else { "plus" }, i + 1);
                    i += 1;
                }
                '=' => {
                    self.push_spaced(if turkish { "eşittir" } else { "equals" }, i + 1);
                    i += 1;
                }
                '&' => {
                    self.push_spaced(if turkish { "ve" } else { "and" }, i + 1);
                    i += 1;
                }
                '@' => {
                    self.push_spaced(if turkish { "et" } else { "at" }, i + 1);
                    i += 1;
                }
                '°' => {
                    self.push_spaced(if turkish { "derece" } else { "degrees" }, i + 1);
                    i += 1;
                }
                _ => {
                    self.out.push(c);
                    i += 1;
                }
            }
        }
        self.out.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Spell out numbers, dates, times, currency, units and abbreviations.
///
/// [`NormLang::Other`] returns the text unchanged.
pub fn expand(text: &str, lang: NormLang) -> String {
    if lang == NormLang::Other {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    Expander {
        chars: &chars,
        lang,
        out: String::with_capacity(text.len() + text.len() / 4),
    }
    .run()
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？')
}

fn is_clause_end(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '—' | '，' | '；' | '：' | '、')
}

/// Split `text` after every character for which `is_end` holds, keeping the
/// delimiter with the piece it closes. ASCII delimiters only split before
/// whitespace, so "3.14" and "e.g." stay whole.
fn split_after(text: &str, is_end: fn(char) -> bool) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((idx, c)) = iter.next() {
        if !is_end(c) {
            continue;
        }
        let next = iter.peek().map(|&(_, n)| n);
        let splits = match next {
            None => false,
            Some(n) => !c.is_ascii() || n.is_whitespace(),
        };
        if splits {
            let end = idx + c.len_utf8();
            pieces.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

/// Break `piece` into parts of at most `max_chars` characters, preferring
/// clause boundaries, then word boundaries, then a hard cut.
fn split_long(piece: &str, max_chars: usize, out: &mut Vec<String>) {
    let piece = piece.trim();
    if piece.is_empty() {
        return;
    }
    if piece.chars().count() <= max_chars {
        out.push(piece.to_string());
        return;
    }

    let clauses = split_after(piece, is_clause_end);
    if clauses.len() > 1 {
        pack(&clauses, max_chars, out);
        return;
    }
    let words: Vec<&str> = piece.split_inclusive(' ').collect();
    if words.len() > 1 {
        pack(&words, max_chars, out);
        return;
    }

    // One unbroken run longer than the limit.
    let chars: Vec<char> = piece.chars().collect();
    for window in chars.chunks(max_chars) {
        out.push(window.iter().collect());
    }
}

/// Greedily join consecutive `parts` into chunks of at most `max_chars`.
fn pack(parts: &[&str], max_chars: usize, out: &mut Vec<String>) {
    let mut current = String::new();
    let mut current_len = 0;
    for part in parts {
        let part_len = part.chars().count();
        if current_len > 0 && current_len + part_len > max_chars {
            split_long(&current, max_chars, out);
            current.clear();
            current_len = 0;
        }
        current.push_str(part);
        current_len += part_len;
    }
    split_long(&current, max_chars, out);
}

/// Split `text` into chunks of at most `max_chars` characters.
///
/// Chunks end at sentence boundaries where possible, then at clause
/// boundaries, then between words. Only an unbroken run longer than
/// `max_chars` is cut mid-word. No chunk is empty.
pub fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let mut out = Vec::new();
    pack(&split_after(text, is_sentence_end), max_chars, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn en(text: &str) -> String {
        normalize(text, NormLang::English, NormalizeMode::Full)
    }

    fn tr(text: &str) -> String {
        normalize(text, NormLang::Turkish, NormalizeMode::Full)
    }

    #[test]
    fn language_tags() {
        assert_eq!(NormLang::from_tag("en"), NormLang::English);
        assert_eq!(NormLang::from_tag("en-US"), NormLang::English);
        assert_eq!(NormLang::from_tag("en_GB"), NormLang::English);
        assert_eq!(NormLang::from_tag("TR"), NormLang::Turkish);
        assert_eq!(NormLang::from_tag("tr_TR"), NormLang::Turkish);
        assert_eq!(NormLang::from_tag("ja"), NormLang::Other);
        assert_eq!(NormLang::from_tag(""), NormLang::Other);
    }

    #[test]
    fn english_table() {
        let cases = [
            // Acceptance table.
            (
                "3 degrees and 25 percent",
                "three degrees and twenty-five percent",
            ),
            ("3° and 25%", "three degrees and twenty-five percent"),
            ("1.5", "one point five"),
            ("2026", "twenty twenty-six"),
            ("%25", "twenty-five percent"),
            ("Dr. Smith is in.", "Doctor Smith is in."),
            // Cardinals.
            ("0", "zero"),
            ("13", "thirteen"),
            ("100", "one hundred"),
            ("101 dalmatians", "one hundred one dalmatians"),
            (
                "1,234,567",
                "one million two hundred thirty-four thousand five hundred sixty-seven",
            ),
            ("10000", "ten thousand"),
            ("3.14", "three point one four"),
            ("0.05", "zero point zero five"),
            ("-5 outside", "minus five outside"),
            ("007", "zero zero seven"),
            // Years.
            ("in 1999.", "in nineteen ninety-nine."),
            ("2000", "two thousand"),
            ("2005", "two thousand five"),
            ("1905", "nineteen oh five"),
            ("1900", "nineteen hundred"),
            ("the 1990s", "the nineteen nineties"),
            ("the 2010s", "the twenty tens"),
            ("3000", "three thousand"),
            // Ordinals.
            ("1st", "first"),
            ("22nd", "twenty-second"),
            ("3rd", "third"),
            ("40th", "fortieth"),
            ("112th", "one hundred twelfth"),
            // Dates and times.
            ("2026-11-12", "November twelfth, twenty twenty-six"),
            ("12.11.2026", "November twelfth, twenty twenty-six"),
            ("11/12/2026", "November twelfth, twenty twenty-six"),
            ("at 3:30", "at three thirty"),
            ("at 10:05", "at ten oh five"),
            ("at 9:00", "at nine o'clock"),
            ("at 9:00 pm", "at nine PM"),
            ("at 3:30pm today", "at three thirty PM today"),
            // Currency.
            ("$5", "five dollars"),
            ("$1", "one dollar"),
            ("$1.50", "one dollar and fifty cents"),
            ("€20", "twenty euros"),
            ("£2.01", "two pounds and one penny"),
            // Units and symbols.
            ("21°C", "twenty-one degrees Celsius"),
            ("1°", "one degree"),
            ("5 km", "five kilometers"),
            ("1kg", "one kilogram"),
            ("60 km/h", "sixty kilometers per hour"),
            ("R&D", "R and D"),
            ("2 + 2 = 4", "two plus two equals four"),
            ("#1 hit", "number one hit"),
            ("B12", "B twelve"),
            // Abbreviations.
            ("Mr. and Mrs. Jones", "Mister and Missus Jones"),
            ("Prof. Lee vs. Dr. Kim", "Professor Lee versus Doctor Kim"),
            ("cats, dogs, etc.", "cats, dogs, et cetera."),
            ("cats etc. and dogs", "cats et cetera and dogs"),
            ("e.g. this", "for example this"),
            ("No. 5", "Number five"),
            ("No. I refuse.", "No. I refuse."),
            ("St. Peter", "Saint Peter"),
            ("on Main St. today", "on Main Street today"),
            // Untouched.
            ("Hello, world!", "Hello, world!"),
            ("The end.", "The end."),
        ];
        for (input, want) in cases {
            assert_eq!(en(input), want, "input: {input:?}");
        }
    }

    #[test]
    fn turkish_table() {
        let cases = [
            // Acceptance table.
            ("3 derece ve yüzde 25", "üç derece ve yüzde yirmi beş"),
            ("3° ve %25", "üç derece ve yüzde yirmi beş"),
            ("1.5", "bir nokta beş"),
            ("1,5", "bir virgül beş"),
            ("2026", "iki bin yirmi altı"),
            ("%25", "yüzde yirmi beş"),
            ("25%", "yüzde yirmi beş"),
            ("Dr. Ayşe geldi.", "Doktor Ayşe geldi."),
            // Cardinals.
            ("0", "sıfır"),
            ("100", "yüz"),
            ("1000", "bin"),
            ("1001", "bin bir"),
            ("1.500", "bin beş yüz"),
            (
                "2.345.678",
                "iki milyon üç yüz kırk beş bin altı yüz yetmiş sekiz",
            ),
            ("1000000", "bir milyon"),
            ("3,14", "üç virgül on dört"),
            ("0,05", "sıfır virgül sıfır beş"),
            ("-5 derece", "eksi beş derece"),
            // Suffixes, dates, times.
            ("2026'da", "iki bin yirmi altıda"),
            ("12.11.2026", "on iki Kasım iki bin yirmi altı"),
            ("12/11/2026", "on iki Kasım iki bin yirmi altı"),
            ("2026-11-12", "on iki Kasım iki bin yirmi altı"),
            ("saat 14:30", "saat on dört otuz"),
            ("saat 09:05", "saat dokuz sıfır beş"),
            ("saat 14:00", "saat on dört"),
            // Currency and units.
            ("₺5", "beş lira"),
            ("5 TL", "beş lira"),
            ("$1,50", "bir dolar elli sent"),
            ("21°C", "yirmi bir derece santigrat"),
            ("5 km", "beş kilometre"),
            // Abbreviations.
            ("Prof. Dr. Kaya", "Profesör Doktor Kaya"),
            ("elma, armut vb.", "elma, armut ve benzeri."),
            ("No. 5", "Numara beş"),
            ("Ar&Ge", "Ar ve Ge"),
        ];
        for (input, want) in cases {
            assert_eq!(tr(input), want, "input: {input:?}");
        }
    }

    #[test]
    fn other_language_is_only_cleaned() {
        assert_eq!(
            normalize("**Hallo** 25%", NormLang::Other, NormalizeMode::Full),
            "Hallo 25%"
        );
    }

    #[test]
    fn clean_only_keeps_digits() {
        assert_eq!(
            normalize(
                "Dr. Who has 3 hearts",
                NormLang::English,
                NormalizeMode::CleanOnly
            ),
            "Dr. Who has 3 hearts"
        );
    }

    #[test]
    fn clean_strips_markup_and_invisibles() {
        assert_eq!(clean("**bold** and `code`"), "bold and code");
        assert_eq!(clean("snake_case"), "snake case");
        assert_eq!(clean("don’t “quote”"), "don't \"quote\"");
        assert_eq!(clean("a\u{200B}b\u{FEFF}c"), "abc");
        assert_eq!(clean("hi \u{1F600} there \u{2764}\u{FE0F}"), "hi there");
        assert_eq!(clean("a\u{00A0}b\tc"), "a b c");
        assert_eq!(
            clean("see [the docs](https://example.com) now"),
            "see the docs now"
        );
        assert_eq!(clean("5–10"), "5-10");
        assert_eq!(clean("  \n \n "), "");
    }

    #[test]
    fn clean_handles_lines() {
        // A wrapped paragraph is one sentence.
        assert_eq!(clean("one\ntwo"), "one two");
        // A paragraph break, heading or list item ends a sentence.
        assert_eq!(clean("one\n\ntwo"), "one. two");
        assert_eq!(clean("done.\n\nnext"), "done. next");
        assert_eq!(clean("# Title\nbody text"), "Title. body text");
        assert_eq!(clean("List:\n- first\n- second"), "List: first. second");
        assert_eq!(clean("line\r\nline"), "line line");
    }

    #[test]
    fn normalize_is_deterministic() {
        let text = "On 2026-11-12 at 9:05, Dr. Lee paid $1,250.75 for 3 items (25% off).";
        assert_eq!(en(text), en(text));
        assert_eq!(
            en(text),
            "On November twelfth, twenty twenty-six at nine oh five, Doctor Lee paid \
             one thousand two hundred fifty dollars and seventy-five cents for three items \
             (twenty-five percent off)."
        );
    }

    #[test]
    fn expansion_leaves_no_digits() {
        for text in [
            "Call 555-1234 or 0532 123 45 67.",
            "v1.2.3 shipped on 1/2/2026 at 23:59:59",
            "99999999999999999999999 is big",
            "x2 3x 4.5.6 7,8 9:99",
        ] {
            let out = en(text);
            assert!(
                !out.chars().any(|c| c.is_ascii_digit()),
                "digits left in {out:?}"
            );
            let out = tr(text);
            assert!(
                !out.chars().any(|c| c.is_ascii_digit()),
                "digits left in {out:?}"
            );
        }
    }

    #[test]
    fn split_text_respects_limit() {
        let sentence = "The quick brown fox jumps over the lazy dog, again and again. ";
        let text = sentence.repeat(100);
        assert!(text.len() > 5000);
        let chunks = split_text(&text, 200);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(!chunk.is_empty());
            assert!(chunk.chars().count() <= 200, "chunk too long: {chunk:?}");
            // Sentences fit, so every chunk ends on a sentence boundary.
            assert!(chunk.ends_with('.'), "chunk cut mid-sentence: {chunk:?}");
        }
        // Nothing but whitespace is lost.
        let rejoined: String = chunks.join(" ");
        assert_eq!(
            rejoined.split_whitespace().collect::<Vec<_>>(),
            text.split_whitespace().collect::<Vec<_>>()
        );
    }

    #[test]
    fn split_text_falls_back_to_clauses_words_and_hard_cuts() {
        // Clause boundary.
        let chunks = split_text("alpha beta gamma, delta epsilon zeta", 20);
        assert_eq!(chunks, vec!["alpha beta gamma,", "delta epsilon zeta"]);
        // Word boundary.
        let chunks = split_text("alpha beta gamma delta", 11);
        assert_eq!(chunks, vec!["alpha beta", "gamma delta"]);
        // Hard cut.
        let chunks = split_text(&"x".repeat(25), 10);
        assert_eq!(
            chunks.iter().map(String::len).collect::<Vec<_>>(),
            [10, 10, 5]
        );
        // Multi-byte text is cut on character boundaries.
        let chunks = split_text(&"ş".repeat(25), 10);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn split_text_keeps_decimals_and_short_text_whole() {
        assert_eq!(
            split_text("Pi is 3.14 exactly.", 100),
            vec!["Pi is 3.14 exactly."]
        );
        assert_eq!(
            split_text("One. Two. Three.", 100),
            vec!["One. Two. Three."]
        );
        assert_eq!(
            split_text("One. Two. Three.", 5),
            vec!["One.", "Two.", "Three", "."]
        );
        assert!(split_text("", 10).is_empty());
        assert!(split_text("   ", 10).is_empty());
    }
}
