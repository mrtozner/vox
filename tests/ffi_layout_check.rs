//! Runtime FFI struct layout verification.
//! Run with: cargo test --test ffi_layout_check --features sherpa

#[cfg(feature = "sherpa")]
mod tests {
    use std::ffi::CStr;
    use std::mem;

    #[test]
    fn check_library_version() {
        unsafe extern "C" {
            fn SherpaOnnxGetVersionStr() -> *const std::os::raw::c_char;
        }
        unsafe {
            let version = SherpaOnnxGetVersionStr();
            if !version.is_null() {
                let v = CStr::from_ptr(version).to_string_lossy();
                eprintln!("sherpa-onnx library version: {v}");
            } else {
                eprintln!("SherpaOnnxGetVersionStr returned null");
            }
        }
    }

    #[test]
    fn check_online_config_offsets() {
        use sherpa_sys::*;

        let cfg = SherpaOnnxOnlineRecognizerConfig::default();
        let base = &cfg as *const _ as usize;

        macro_rules! check_offset {
            ($field:ident, $expected:expr) => {
                let actual = &cfg.$field as *const _ as usize - base;
                eprintln!(
                    "  {}: offset {} (expected {}): {}",
                    stringify!($field),
                    actual,
                    $expected,
                    if actual == $expected {
                        "OK"
                    } else {
                        "MISMATCH!"
                    }
                );
                assert_eq!(
                    actual,
                    $expected,
                    "offset mismatch for {}",
                    stringify!($field)
                );
            };
        }

        eprintln!(
            "SherpaOnnxOnlineRecognizerConfig size: {}",
            mem::size_of::<SherpaOnnxOnlineRecognizerConfig>()
        );
        check_offset!(feat_config, 0);
        check_offset!(model_config, 8);
        check_offset!(decoding_method, 144);
        check_offset!(max_active_paths, 152);
        check_offset!(enable_endpoint, 156);
        check_offset!(rule1_min_trailing_silence, 160);
        check_offset!(rule2_min_trailing_silence, 164);
        check_offset!(rule3_min_utterance_length, 168);
        check_offset!(hotwords_file, 176);
        check_offset!(hotwords_score, 184);
        check_offset!(ctc_fst_decoder_config, 192);
        check_offset!(rule_fsts, 208);
        check_offset!(rule_fars, 216);
        check_offset!(blank_penalty, 224);
        check_offset!(hotwords_buf, 232);
        check_offset!(hotwords_buf_size, 240);
        check_offset!(hr, 248);
    }

    #[test]
    fn check_online_model_config_offsets() {
        use sherpa_sys::*;

        let cfg = SherpaOnnxOnlineModelConfig::default();
        let base = &cfg as *const _ as usize;

        macro_rules! check_offset {
            ($field:ident, $expected:expr) => {
                let actual = &cfg.$field as *const _ as usize - base;
                eprintln!(
                    "  {}: offset {} (expected {}): {}",
                    stringify!($field),
                    actual,
                    $expected,
                    if actual == $expected {
                        "OK"
                    } else {
                        "MISMATCH!"
                    }
                );
                assert_eq!(
                    actual,
                    $expected,
                    "offset mismatch for {}",
                    stringify!($field)
                );
            };
        }

        eprintln!(
            "SherpaOnnxOnlineModelConfig size: {}",
            mem::size_of::<SherpaOnnxOnlineModelConfig>()
        );
        check_offset!(transducer, 0);
        check_offset!(paraformer, 24);
        check_offset!(zipformer2_ctc, 40);
        check_offset!(tokens, 48);
        check_offset!(num_threads, 56);
        check_offset!(provider, 64);
        check_offset!(debug, 72);
        check_offset!(model_type, 80);
        check_offset!(modeling_unit, 88);
        check_offset!(bpe_vocab, 96);
        check_offset!(tokens_buf, 104);
        check_offset!(tokens_buf_size, 112);
        check_offset!(nemo_ctc, 120);
    }

    #[test]
    fn check_offline_model_config_offsets() {
        use sherpa_sys::*;

        let cfg = SherpaOnnxOfflineModelConfig::default();
        let base = &cfg as *const _ as usize;

        macro_rules! check_offset {
            ($field:ident, $expected:expr) => {
                let actual = &cfg.$field as *const _ as usize - base;
                eprintln!(
                    "  {}: offset {} (expected {}): {}",
                    stringify!($field),
                    actual,
                    $expected,
                    if actual == $expected {
                        "OK"
                    } else {
                        "MISMATCH!"
                    }
                );
                assert_eq!(
                    actual,
                    $expected,
                    "offset mismatch for {}",
                    stringify!($field)
                );
            };
        }

        eprintln!(
            "SherpaOnnxOfflineModelConfig size: {}",
            mem::size_of::<SherpaOnnxOfflineModelConfig>()
        );
        assert_eq!(mem::size_of::<SherpaOnnxOfflineModelConfig>(), 392);
        check_offset!(transducer, 0);
        check_offset!(paraformer, 24);
        check_offset!(nemo_ctc, 32);
        check_offset!(whisper, 40);
        check_offset!(tdnn, 88);
        check_offset!(tokens, 96);
        check_offset!(num_threads, 104);
        check_offset!(debug, 108);
        check_offset!(provider, 112);
        check_offset!(model_type, 120);
        check_offset!(modeling_unit, 128);
        check_offset!(bpe_vocab, 136);
        check_offset!(telespeech_ctc, 144);
        check_offset!(sense_voice, 152);
        check_offset!(moonshine, 176);
        check_offset!(fire_red_asr, 208);
        check_offset!(dolphin, 224);
        check_offset!(zipformer_ctc, 232);
        check_offset!(canary, 240);
        check_offset!(wenet_ctc, 280);
        check_offset!(omnilingual, 288);
        check_offset!(medasr, 296);
        check_offset!(funasr_nano, 304);
    }

    #[test]
    fn check_offline_result_offsets() {
        use sherpa_sys::*;

        eprintln!(
            "SherpaOnnxOfflineRecognizerResult size: {}",
            mem::size_of::<SherpaOnnxOfflineRecognizerResult>()
        );
        assert_eq!(mem::size_of::<SherpaOnnxOfflineRecognizerResult>(), 128);
    }

    #[test]
    fn try_create_online_recognizer_with_empty_config() {
        use sherpa_sys::*;

        // Call with default (all-null) config — should return null, NOT crash
        let cfg = SherpaOnnxOnlineRecognizerConfig::default();
        eprintln!("About to call SherpaOnnxCreateOnlineRecognizer with default config...");
        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&cfg) };
        eprintln!("Returned: {:?} (null={})", recognizer, recognizer.is_null());
        // We expect null (no valid model paths), but we should NOT crash
        assert!(
            recognizer.is_null(),
            "expected null recognizer with empty config"
        );
    }

    #[test]
    fn try_create_online_recognizer_with_real_model() {
        use sherpa_sys::*;
        use std::ffi::CString;
        use std::path::PathBuf;

        let model_dir = PathBuf::from("/tmp/sherpa-test2");

        if !model_dir.exists() {
            eprintln!("Skipping: model dir not found at {}", model_dir.display());
            return;
        }

        let encoder_path = model_dir.join("encoder.int8.onnx");
        let decoder_path = model_dir.join("decoder.int8.onnx");
        let joiner_path = model_dir.join("joiner.int8.onnx");
        let tokens_path = model_dir.join("tokens.txt");

        if !encoder_path.exists()
            || !decoder_path.exists()
            || !joiner_path.exists()
            || !tokens_path.exists()
        {
            eprintln!("Skipping: not all model files found");
            return;
        }

        // Keep CStrings alive for the duration of the call
        let encoder_cs = CString::new(encoder_path.to_str().unwrap()).unwrap();
        let decoder_cs = CString::new(decoder_path.to_str().unwrap()).unwrap();
        let joiner_cs = CString::new(joiner_path.to_str().unwrap()).unwrap();
        let tokens_cs = CString::new(tokens_path.to_str().unwrap()).unwrap();
        let provider_cs = CString::new("cpu").unwrap();
        let decoding_cs = CString::new("greedy_search").unwrap();

        eprintln!("encoder: {}", encoder_path.display());
        eprintln!("decoder: {}", decoder_path.display());
        eprintln!("joiner: {}", joiner_path.display());
        eprintln!("tokens: {}", tokens_path.display());

        // Use zeroed() instead of Default — matches C's memset(0)
        let mut cfg: SherpaOnnxOnlineRecognizerConfig = unsafe { std::mem::zeroed() };
        cfg.feat_config.sample_rate = 16000;
        cfg.feat_config.feature_dim = 80;
        cfg.model_config.transducer.encoder = encoder_cs.as_ptr();
        cfg.model_config.transducer.decoder = decoder_cs.as_ptr();
        cfg.model_config.transducer.joiner = joiner_cs.as_ptr();
        cfg.model_config.tokens = tokens_cs.as_ptr();
        cfg.model_config.num_threads = 4;
        cfg.model_config.provider = provider_cs.as_ptr();
        cfg.model_config.debug = 0;
        cfg.decoding_method = decoding_cs.as_ptr();
        cfg.enable_endpoint = 1;
        cfg.rule1_min_trailing_silence = 2.4;
        cfg.rule2_min_trailing_silence = 1.2;
        cfg.rule3_min_utterance_length = 20.0;

        eprintln!("About to call SherpaOnnxCreateOnlineRecognizer with real model...");
        let recognizer = unsafe { SherpaOnnxCreateOnlineRecognizer(&cfg) };
        eprintln!("Returned: {:?} (null={})", recognizer, recognizer.is_null());

        if !recognizer.is_null() {
            eprintln!("SUCCESS! Recognizer created.");
            unsafe {
                SherpaOnnxDestroyOnlineRecognizer(recognizer);
            }
        } else {
            eprintln!("Recognizer creation returned null (model issue?)");
        }
    }
}
