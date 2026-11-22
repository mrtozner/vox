fn main() {
    const SRC: &str = "src/transcription/en_ipa.c";
    cc::Build::new().file(SRC).compile("es");
    println!("cargo:rerun-if-changed={}", SRC);
}
