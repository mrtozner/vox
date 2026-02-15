class Vox < Formula
  desc "Open-source local voice AI framework - speech-to-text and text-to-speech"
  homepage "https://github.com/mrtozner/vox"
  url "https://github.com/mrtozner/vox/archive/refs/tags/v0.2.0.tar.gz"
  # sha256 will be set after v0.2.0 tag is created
  sha256 "PLACEHOLDER"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/mrtozner/vox.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", "--path", ".", "--root", prefix,
           "--features", "cli", "--locked"
  end

  def caveats
    <<~EOS
      To get started, download the required voice models:

        vox models download silero-vad
        vox models download whisper-tiny.en

      Then start transcribing:

        vox listen

      For text-to-speech, install with kokoro support:

        cargo install vox --features cli,kokoro
        vox models download kokoro
        vox models download kokoro-voices
        vox speak "Hello from Vox!"

      For voice chat with an LLM (requires Ollama):

        vox chat --llm llama3.2
    EOS
  end

  test do
    assert_match "vox #{version}", shell_output("#{bin}/vox --version")
  end
end
