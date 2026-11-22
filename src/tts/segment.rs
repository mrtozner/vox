//! Shared text segmenter and TTS sanitizer.
//!
//! One splitter for every path that turns text into speech: the streaming
//! LLM→TTS path (`streaming_chat`) feeds tokens into a
//! [`Segmenter`], and the batch path ([`super::SentenceStreamingAdapter`])
//! calls [`split_sentences`], which runs the same code.
//!
//! Segmentation rules:
//!
//! - The **first** segment of a reply is cut early, at a clause boundary
//!   (`,` `;` `:`) once it has about [`FIRST_CHUNK_MIN_WORDS`] words, so the
//!   first audio starts quickly.
//! - Later segments end only at a real sentence end (`.` `!` `?` `…` or a line
//!   break), with a ceiling of about [`MAX_SEGMENT_WORDS`] words.
//! - `;` is a clause boundary, not a sentence end.
//! - Abbreviations ("Dr. Smith", "e.g. this"), initials ("J. K. Rowling"),
//!   list numbers ("1. First") and decimals ("3.50") do not end a segment.
//!
//! Every emitted segment has been passed through [`sanitize_for_tts`], so the
//! string given to the TTS backend is the same string a caller stores in
//! conversation memory. The raw text is not altered here; callers that display
//! text keep their own copy (see the `on_token` callback in `streaming_chat`).

/// Word count after which the first segment may be cut at a clause boundary.
pub const FIRST_CHUNK_MIN_WORDS: usize = 8;

/// Word ceiling for a segment that has no sentence end.
pub const MAX_SEGMENT_WORDS: usize = 30;

/// Abbreviations whose trailing period never ends a sentence.
///
/// Abbreviations that often do end a sentence ("etc.", "p.m.", "Inc.") are
/// deliberately left out: they are handled by the rule that a period followed
/// by a lowercase word is not a sentence end.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "e.g", "i.e", "eg", "ie", "fig", "approx", "mt",
    "gen", "col", "lt", "sgt", "capt", "cmdr", "rev", "hon", "gov", "sen", "rep", "dept", "vol",
    "cf", "ca",
];

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Sentence terminators that need no following whitespace.
fn is_cjk_terminator(c: char) -> bool {
    matches!(c, '。' | '！' | '？')
}

/// Characters that may follow a terminator and still belong to the sentence.
fn is_closer(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | ')' | ']' | '}' | '”' | '’' | '»' | '*' | '_' | '`'
    )
}

fn is_opener(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '(' | '[' | '{' | '“' | '‘' | '«' | '*' | '_' | '`'
    )
}

/// True when the period at `dot` belongs to an abbreviation, an initial or a
/// list number rather than ending a sentence.
fn period_is_not_sentence_end(buf: &[char], dot: usize) -> bool {
    let mut start = dot;
    while start > 0 && !buf[start - 1].is_whitespace() && !is_opener(buf[start - 1]) {
        start -= 1;
    }
    let word: String = buf[start..dot].iter().collect();
    if word.is_empty() {
        return false;
    }
    let lower = word.to_lowercase();
    if ABBREVIATIONS.contains(&lower.as_str()) {
        return true;
    }
    // Initials: "J." or the last part of "U.S."
    let last_part = word.rsplit('.').next().unwrap_or("");
    let mut lp = last_part.chars();
    if let (Some(c), None) = (lp.next(), lp.next())
        && c.is_uppercase()
    {
        return true;
    }
    // List numbers: "1. First", "Steps: 2. Then".
    if word.len() <= 3 && word.chars().all(|c| c.is_ascii_digit()) {
        let before = buf[..start]
            .iter()
            .rev()
            .find(|c| !c.is_whitespace() && !matches!(c, '*' | '#' | '-' | '>' | '_'));
        return matches!(before, None | Some(':'));
    }
    false
}

/// Find the end (exclusive) of the next segment in `buf`, if one is complete.
///
/// `first` enables the early clause cut. `fin` means no more text will come,
/// so an end of buffer counts as confirmation of a trailing terminator.
fn find_cut(buf: &[char], first: bool, fin: bool) -> Option<usize> {
    let len = buf.len();
    let mut words = 0usize;
    let mut in_word = false;
    let mut seen_content = false;
    // Last clause boundary seen, with the word count before it.
    let mut last_clause: Option<(usize, usize)> = None;
    let mut i = 0;

    while i < len {
        let c = buf[i];

        if c.is_whitespace() {
            if in_word {
                in_word = false;
                if words >= MAX_SEGMENT_WORDS {
                    return Some(match last_clause {
                        Some((cut, w)) if w >= 4 => cut,
                        _ => i,
                    });
                }
            }
            if c == '\n' && seen_content {
                return Some(i + 1);
            }
            i += 1;
            continue;
        }

        if !in_word {
            in_word = true;
            words += 1;
        }
        seen_content = true;

        if is_cjk_terminator(c) {
            let mut j = i + 1;
            while j < len && (is_cjk_terminator(buf[j]) || is_closer(buf[j])) {
                j += 1;
            }
            return Some(j);
        }

        if is_terminator(c) {
            let mut j = i + 1;
            while j < len && is_terminator(buf[j]) {
                j += 1;
            }
            let single_period = c == '.' && j == i + 1;
            let soft = c == '.' || c == '…';
            while j < len && is_closer(buf[j]) {
                j += 1;
            }
            if j >= len {
                // Nothing after the terminator yet: "3." may become "3.50".
                return if fin { Some(len) } else { None };
            }
            if !buf[j].is_whitespace() {
                i = j;
                continue;
            }
            if single_period && period_is_not_sentence_end(buf, i) {
                i = j;
                continue;
            }
            if soft {
                // A period followed by a lowercase word is an abbreviation
                // or an ellipsis mid-sentence, not a sentence end.
                let mut k = j;
                while k < len && buf[k].is_whitespace() && buf[k] != '\n' {
                    k += 1;
                }
                if k >= len {
                    return if fin { Some(len) } else { None };
                }
                if buf[k].is_lowercase() {
                    i = j;
                    continue;
                }
            }
            return Some(j);
        }

        if matches!(c, ',' | ';' | ':') && i + 1 < len && buf[i + 1].is_whitespace() {
            if first && words >= FIRST_CHUNK_MIN_WORDS {
                return Some(i + 1);
            }
            last_clause = Some((i + 1, words));
        }

        i += 1;
    }

    if fin && seen_content { Some(len) } else { None }
}

/// Region of the token stream that is never spoken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hidden {
    None,
    Think,
    Code,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CODE_FENCE: &str = "```";

/// Incremental sentence segmenter for a token stream.
///
/// Push tokens as they arrive; each call returns the segments completed so
/// far, already sanitized for TTS. Call [`Segmenter::flush`] at end of stream.
#[derive(Debug)]
pub struct Segmenter {
    /// Raw text not yet examined for hidden regions (may end in a partial tag).
    pending: String,
    hidden: Hidden,
    buf: Vec<char>,
    first: bool,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Segmenter {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            hidden: Hidden::None,
            buf: Vec::new(),
            first: true,
        }
    }

    /// Add a token and return any segments it completed (usually 0 or 1).
    pub fn push(&mut self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }
        self.pending.push_str(token);
        self.filter_hidden(false);
        self.drain(false)
    }

    /// End of stream: return whatever speakable text is left.
    pub fn flush(&mut self) -> Vec<String> {
        self.filter_hidden(true);
        self.drain(true)
    }

    /// Move visible text from `pending` to `buf`, dropping `<think>` blocks
    /// and fenced code. A possible partial delimiter at the end of `pending`
    /// is held back until the next token decides it.
    fn filter_hidden(&mut self, fin: bool) {
        loop {
            match self.hidden {
                Hidden::None => {
                    let think = self.pending.find(THINK_OPEN);
                    let code = self.pending.find(CODE_FENCE);
                    let (pos, delim, next) = match (think, code) {
                        (Some(t), Some(c)) if c < t => (c, CODE_FENCE, Hidden::Code),
                        (Some(t), _) => (t, THINK_OPEN, Hidden::Think),
                        (None, Some(c)) => (c, CODE_FENCE, Hidden::Code),
                        (None, None) => {
                            let keep = if fin {
                                0
                            } else {
                                partial_suffix(&self.pending, &[THINK_OPEN, CODE_FENCE])
                            };
                            let cut = self.pending.len() - keep;
                            self.buf.extend(self.pending[..cut].chars());
                            self.pending.drain(..cut);
                            return;
                        }
                    };
                    self.buf.extend(self.pending[..pos].chars());
                    // A hidden block separates what is around it.
                    self.buf.push('\n');
                    self.pending.drain(..pos + delim.len());
                    self.hidden = next;
                }
                Hidden::Think | Hidden::Code => {
                    let close = if self.hidden == Hidden::Think {
                        THINK_CLOSE
                    } else {
                        CODE_FENCE
                    };
                    match self.pending.find(close) {
                        Some(pos) => {
                            self.pending.drain(..pos + close.len());
                            self.hidden = Hidden::None;
                        }
                        None => {
                            let keep = if fin {
                                0
                            } else {
                                partial_suffix(&self.pending, &[close])
                            };
                            let cut = self.pending.len() - keep;
                            self.pending.drain(..cut);
                            return;
                        }
                    }
                }
            }
        }
    }

    fn drain(&mut self, fin: bool) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(cut) = find_cut(&self.buf, self.first, fin) {
            let raw: String = self.buf[..cut].iter().collect();
            let mut rest = cut;
            while rest < self.buf.len() && self.buf[rest].is_whitespace() {
                rest += 1;
            }
            self.buf.drain(..rest);
            let clean = sanitize_for_tts(&raw);
            if is_speakable(&clean) {
                out.push(clean);
                self.first = false;
            }
        }
        if fin {
            self.buf.clear();
        }
        out
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of one of
/// `delims` (so it could still grow into that delimiter).
fn partial_suffix(text: &str, delims: &[&str]) -> usize {
    let mut best = 0;
    for delim in delims {
        for n in (1..delim.len()).rev() {
            if n <= text.len()
                && text.is_char_boundary(text.len() - n)
                && text.ends_with(&delim[..n])
            {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

fn is_speakable(text: &str) -> bool {
    text.chars().any(|c| c.is_alphanumeric())
}

/// Split a complete text into TTS segments using the shared [`Segmenter`].
///
/// Segments are sanitized; text with nothing speakable yields an empty list.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut seg = Segmenter::new();
    let mut out = seg.push(text);
    out.extend(seg.flush());
    out
}

fn is_emoji(c: char) -> bool {
    matches!(
        c as u32,
        0x1F000..=0x1FAFF      // pictographs, emoticons, transport, flags
            | 0x2600..=0x27BF  // misc symbols, dingbats
            | 0x2B00..=0x2BFF  // stars, arrows
            | 0x2300..=0x23FF  // technical (watch, hourglass)
            | 0xFE00..=0xFE0F  // variation selectors
            | 0xE0020..=0xE007F // tag characters
            | 0x200D           // zero-width joiner
            | 0x20E3 // keycap
    )
}

/// Remove a delimited region, e.g. `<think>…</think>`. An unclosed region
/// runs to the end of the text.
fn strip_region(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        out.push('\n');
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => rest = &after[end + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Rewrite `[text](url)` and `![alt](url)` to their text, and drop bare URLs.
fn strip_links(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let bracket = if c == '[' {
            Some(i)
        } else if c == '!' && chars.get(i + 1) == Some(&'[') {
            Some(i + 1)
        } else {
            None
        };
        if let Some(open) = bracket
            && let Some(close) = (open + 1..chars.len()).find(|&k| chars[k] == ']')
            && chars.get(close + 1) == Some(&'(')
            && let Some(end) = (close + 2..chars.len()).find(|&k| chars[k] == ')')
        {
            out.extend(&chars[open + 1..close]);
            i = end + 1;
            continue;
        }
        let at_word_start = i == 0 || !chars[i - 1].is_alphanumeric();
        if at_word_start {
            let rest: String = chars[i..chars.len().min(i + 8)].iter().collect();
            if rest.starts_with("http://")
                || rest.starts_with("https://")
                || rest.starts_with("www.")
            {
                let mut end = i;
                while end < chars.len() && !chars[end].is_whitespace() {
                    end += 1;
                }
                // Keep sentence punctuation that trails the URL.
                while end > i && matches!(chars[end - 1], '.' | ',' | '!' | '?' | ';' | ':' | ')') {
                    end -= 1;
                }
                i = end;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Strip markdown markers at the start of a line (headings, quotes, bullets).
fn strip_line_prefix(line: &str) -> &str {
    let mut s = line.trim_start();
    loop {
        let before = s;
        s = s.trim_start_matches('#').trim_start();
        if let Some(rest) = s.strip_prefix('>') {
            s = rest.trim_start();
        }
        for bullet in ["- ", "* ", "+ ", "• "] {
            if let Some(rest) = s.strip_prefix(bullet) {
                s = rest.trim_start();
            }
        }
        if s == before {
            return s;
        }
    }
}

/// Turn LLM output into text that is safe to hand to a TTS backend.
///
/// Removes `<think>` blocks, fenced code, markdown markup, URLs and emoji, and
/// collapses whitespace. Link text, inline code content and list numbers are
/// kept. The result may be empty.
pub fn sanitize_for_tts(text: &str) -> String {
    let text = strip_region(text, THINK_OPEN, THINK_CLOSE);
    let text = strip_region(&text, CODE_FENCE, CODE_FENCE);
    let text = strip_links(&text);

    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let line = strip_line_prefix(line);
        // Horizontal rules and table separator rows carry no words.
        if !line.is_empty()
            && line
                .chars()
                .all(|c| matches!(c, '-' | '=' | '|' | ':' | ' ' | '_'))
        {
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            match c {
                '*' | '`' | '~' => {}
                '|' => out.push(' '),
                '_' => {
                    // Keep snake_case; drop emphasis underscores.
                    let prev = i.checked_sub(1).map(|p| chars[p]);
                    let next = chars.get(i + 1).copied();
                    let inner = prev.is_some_and(|p| p.is_alphanumeric())
                        && next.is_some_and(|n| n.is_alphanumeric());
                    if inner {
                        out.push(' ');
                    }
                }
                c if is_emoji(c) => {}
                c => out.push(c),
            }
        }
        out.push(' ');
    }

    // Collapse whitespace; punctuation left alone by a removed URL or emoji
    // goes back onto the previous word.
    let mut clean = String::with_capacity(out.len());
    for word in out.split_whitespace() {
        let orphan = word
            .chars()
            .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':'));
        if !clean.is_empty() && !orphan {
            clean.push(' ');
        }
        clean.push_str(word);
    }
    clean
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `text` word by word, the way an LLM stream delivers it.
    fn stream(text: &str) -> Vec<String> {
        let mut seg = Segmenter::new();
        let mut out = Vec::new();
        let mut token = String::new();
        for c in text.chars() {
            if c.is_whitespace() && !token.is_empty() && !token.ends_with(char::is_whitespace) {
                out.extend(seg.push(&token));
                token.clear();
            }
            token.push(c);
        }
        out.extend(seg.push(&token));
        out.extend(seg.flush());
        out
    }

    /// Batch and streaming must agree on every input.
    fn both(text: &str) -> Vec<String> {
        let batch = split_sentences(text);
        assert_eq!(batch, stream(text), "batch and stream differ for {text:?}");
        batch
    }

    #[test]
    fn splits_plain_sentences() {
        assert_eq!(both("Hello. World."), vec!["Hello.", "World."]);
        assert_eq!(
            both("Hello! How are you? Fine."),
            vec!["Hello!", "How are you?", "Fine."]
        );
        assert_eq!(both("Hello. World"), vec!["Hello.", "World"]);
    }

    #[test]
    fn empty_and_whitespace_yield_nothing() {
        assert!(both("").is_empty());
        assert!(both("   \n ").is_empty());
        assert!(both("...").is_empty());
    }

    #[test]
    fn sentence_is_emitted_only_after_lookahead() {
        let mut seg = Segmenter::new();
        assert!(seg.push("Hello").is_empty());
        assert!(seg.push(".").is_empty());
        assert!(seg.push(" ").is_empty());
        assert_eq!(seg.push("World"), vec!["Hello."]);
        assert!(seg.push("!").is_empty());
        assert_eq!(seg.flush(), vec!["World!"]);
    }

    #[test]
    fn decimals_stay_together() {
        assert_eq!(
            both("The price is 3.50 dollars."),
            vec!["The price is 3.50 dollars."]
        );
        // The decimal point arrives at a token boundary.
        let mut seg = Segmenter::new();
        assert!(seg.push("It costs 3.").is_empty());
        assert!(seg.push("50 dollars.").is_empty());
        assert_eq!(seg.flush(), vec!["It costs 3.50 dollars."]);
    }

    #[test]
    fn mid_length_sentence_is_one_segment() {
        // 13, 16 and 20 words, no clause punctuation: never split, in first
        // position or later.
        let s13 = "The quick brown fox jumps over the lazy dog near the river bank.";
        let s16 = "My neighbour walks her two dogs around the old park every single morning before work starts.";
        let s20 = "We should probably leave a little earlier tomorrow because the traffic on the bridge gets really bad after eight o'clock.";
        for s in [s13, s16, s20] {
            assert_eq!(both(s), vec![s]);
            assert_eq!(both(&format!("Okay. {s}")), vec!["Okay.", s]);
        }
    }

    #[test]
    fn later_sentences_ignore_clause_boundaries() {
        let s = "After the meeting ended, we walked to the station, bought two tickets, and waited for the late train.";
        assert_eq!(s.split_whitespace().count(), 18);
        assert_eq!(both(&format!("Sure. {s}")), vec!["Sure.", s]);
    }

    #[test]
    fn first_segment_flushes_early_at_clause() {
        let out = both(
            "Well the short answer to your question is yes, but there are a few details worth knowing. Second sentence.",
        );
        assert_eq!(
            out,
            vec![
                "Well the short answer to your question is yes,",
                "but there are a few details worth knowing.",
                "Second sentence."
            ]
        );
    }

    #[test]
    fn early_flush_needs_enough_words() {
        // The comma comes after three words: too early to cut.
        assert_eq!(
            both("Yes, of course, I can help with that."),
            vec!["Yes, of course, I can help with that."]
        );
    }

    #[test]
    fn semicolon_is_not_a_sentence_end() {
        assert_eq!(
            both("Fine. I came; I saw; I left."),
            vec!["Fine.", "I came; I saw; I left."]
        );
    }

    #[test]
    fn abbreviations_do_not_split() {
        assert_eq!(both("Dr. Smith said hello."), vec!["Dr. Smith said hello."]);
        assert_eq!(
            both("Use a tool, e.g. a hammer. Then stop."),
            vec!["Use a tool, e.g. a hammer.", "Then stop."]
        );
        assert_eq!(
            both("Mr. and Mrs. Jones met Prof. Lee."),
            vec!["Mr. and Mrs. Jones met Prof. Lee."]
        );
        assert_eq!(
            both("J. K. Rowling wrote it. I read it."),
            vec!["J. K. Rowling wrote it.", "I read it."]
        );
        assert_eq!(
            both("We sell pens, paper etc. and more. Done."),
            vec!["We sell pens, paper etc. and more.", "Done."]
        );
    }

    #[test]
    fn abbreviation_split_across_tokens() {
        let mut seg = Segmenter::new();
        assert!(seg.push("Dr").is_empty());
        assert!(seg.push(".").is_empty());
        assert!(seg.push(" Smith").is_empty());
        assert!(seg.push(" is in.").is_empty());
        assert_eq!(seg.flush(), vec!["Dr. Smith is in."]);
    }

    #[test]
    fn list_numbers_do_not_split() {
        assert_eq!(both("1. First"), vec!["1. First"]);
        assert_eq!(
            both("1. First do this. 2. Then do that."),
            vec!["1. First do this.", "2. Then do that."]
        );
        assert_eq!(both("1. First\n2. Second\n"), vec!["1. First", "2. Second"]);
        // A number that ends a sentence still ends it.
        assert_eq!(
            both("I have 3. You have 4."),
            vec!["I have 3.", "You have 4."]
        );
    }

    #[test]
    fn long_run_on_is_capped() {
        let words: Vec<String> = (0..70).map(|i| format!("word{i}")).collect();
        let out = both(&words.join(" "));
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].split_whitespace().count(), MAX_SEGMENT_WORDS);
        assert_eq!(out[1].split_whitespace().count(), MAX_SEGMENT_WORDS);
        assert_eq!(out.join(" "), words.join(" "));
    }

    #[test]
    fn cap_prefers_clause_boundary() {
        let head = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon,";
        let tail: Vec<String> = (0..15).map(|i| format!("w{i}")).collect();
        let text = format!("Hi. {head} {}", tail.join(" "));
        let out = both(&text);
        assert_eq!(out[0], "Hi.");
        assert_eq!(out[1], head);
    }

    #[test]
    fn think_block_is_never_spoken() {
        assert_eq!(
            both("<think>The user greets me. I should reply.</think>Hello there. How are you?"),
            vec!["Hello there.", "How are you?"]
        );
        // Tag split across tokens.
        let mut seg = Segmenter::new();
        let mut out = Vec::new();
        for t in ["<th", "ink>secret. ", "plan</th", "ink>", " Hi", " there."] {
            out.extend(seg.push(t));
        }
        out.extend(seg.flush());
        assert_eq!(out, vec!["Hi there."]);
        // Unclosed block hides everything after it.
        assert_eq!(both("Okay. <think>never closed. At all."), vec!["Okay."]);
    }

    #[test]
    fn lone_angle_bracket_is_not_swallowed() {
        assert_eq!(both("Use a < b here."), vec!["Use a < b here."]);
        let mut seg = Segmenter::new();
        assert!(seg.push("x <").is_empty());
        assert_eq!(seg.flush(), vec!["x <"]);
    }

    #[test]
    fn code_fence_is_skipped() {
        assert_eq!(
            both(
                "Run this:\n```rust\nfn main() { println!(\"hi. there\"); }\n```\nThen you are done."
            ),
            vec!["Run this:", "Then you are done."]
        );
    }

    #[test]
    fn markdown_is_stripped_from_segments() {
        assert_eq!(
            both("## Title\n**Bold** and _soft_ text with `code`.\n- item one\n- item two"),
            vec![
                "Title",
                "Bold and soft text with code.",
                "item one",
                "item two"
            ]
        );
    }

    #[test]
    fn sanitize_removes_markup_urls_and_emoji() {
        assert_eq!(
            sanitize_for_tts(
                "See [the docs](https://example.com/a.b) or https://example.com/x. 🎉 Done ✅"
            ),
            "See the docs or. Done"
        );
        assert_eq!(sanitize_for_tts("**hi** ~~no~~ `x`"), "hi no x");
        assert_eq!(sanitize_for_tts("snake_case stays"), "snake case stays");
        assert_eq!(
            sanitize_for_tts("| a | b |\n|---|---|\n| 1 | 2 |"),
            "a b 1 2"
        );
        assert_eq!(sanitize_for_tts("<think>hidden</think>shown"), "shown");
        assert_eq!(sanitize_for_tts("> quoted\n\n---\n# Head"), "quoted Head");
        assert_eq!(sanitize_for_tts("🎉✨"), "");
        assert_eq!(
            sanitize_for_tts("Türkçe çalışıyor, değil mi?"),
            "Türkçe çalışıyor, değil mi?"
        );
    }

    #[test]
    fn emoji_only_segment_is_dropped() {
        assert_eq!(both("Great! 🎉 Next one."), vec!["Great!", "Next one."]);
    }

    #[test]
    fn cjk_terminators_split_without_whitespace() {
        assert_eq!(both("你好。再见！"), vec!["你好。", "再见！"]);
    }
}
