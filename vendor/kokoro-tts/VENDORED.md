# Vendored kokoro-tts 0.3.2

Copied from crates.io `kokoro-tts` 0.3.2 (Apache-2.0, see `LICENSE`). Examples
were left out.

Local change:

- `src/transcription/en_ipa.c`: `ph_list2` is `static`. Upstream (0.3.2 and
  0.3.3) leaves it global, so it collides with espeak-ng's own `ph_list2` when
  a binary links both Kokoro and Piper: lld (the default Linux linker since
  Rust 1.100) rejects the duplicate symbol, and linkers that accept it make
  both engines share one phoneme buffer.

The phonemizer in `en_ipa.c` and `dict/espeak.dict` appear to be derived from
espeak-ng (GPL-3.0-or-later); see the repository's `NOTICE`.
