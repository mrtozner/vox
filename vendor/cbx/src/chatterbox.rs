use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use half::f16;
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::tensor::TensorElementType;
use ort::value::{DynValue, Tensor, ValueType};
use tokenizers::Tokenizer;

use crate::audio::TARGET_SAMPLE_RATE;
use crate::voice::{TensorF32, TensorI64, VoiceProfile};
use crate::{audio, hf};

// The model card snippet uses 6562 for STOP, but the exported models in this HF repo have an
// embedding table sized so that the maximum valid token id is `vocab_size - 1` (e.g. 6560).
// We derive STOP dynamically from the language model's logits vector length.
const SILENCE_TOKEN: i64 = 4299;

// These are baked into the published inference snippet for this repo.
const DEFAULT_NUM_KV_HEADS: usize = 16;
const DEFAULT_HEAD_DIM: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElemType {
    F16,
    F32,
}

fn elem_type_of_value_type(vt: &ValueType) -> Option<ElemType> {
    let ValueType::Tensor { ty, .. } = vt else {
        return None;
    };
    match ty {
        ort::tensor::TensorElementType::Float16 => Some(ElemType::F16),
        ort::tensor::TensorElementType::Float32 => Some(ElemType::F32),
        _ => None,
    }
}

fn elem_type_of_named_input(session: &Session, name: &str) -> Option<ElemType> {
    session
        .inputs()
        .iter()
        .find(|i| i.name() == name)
        .and_then(|i| elem_type_of_value_type(&i.dtype()))
}

fn tensor_from_f32_profile(t: &TensorF32, out_type: ElemType) -> Result<DynValue> {
    match out_type {
        ElemType::F32 => Ok(Tensor::from_array((t.shape.clone(), t.data.clone()))
            .context("tensor from f32 profile")?
            .into_dyn()),
        ElemType::F16 => {
            let data16: Vec<f16> = t.data.iter().copied().map(f16::from_f32).collect();
            Ok(Tensor::from_array((t.shape.clone(), data16))
                .context("tensor from f16 profile")?
                .into_dyn())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Auto,
    Cpu,
    CoreML,
    DirectML,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub parallel_execution: bool,
    pub execution_provider: ExecutionProvider,
    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub coreml_cache_dir: Option<PathBuf>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
            execution_provider: ExecutionProvider::Auto,
            coreml_cache_dir: None,
        }
    }
}

#[derive(Debug)]
pub struct Chatterbox {
    tokenizer: Tokenizer,
    conditional_decoder: Session,
    speech_encoder: Session,
    embed_tokens: Session,
    language_model: Session,

    speech_encoder_audio_input: String,
    speech_encoder_attention_mask_input: Option<String>,

    lm_inputs_embeds_type: ElemType,
    kv_type: ElemType,
    kv_shape: (usize, usize, usize, usize), // [B, H, T, D] with T=0 for init
    past_kv_names: Vec<String>,
    present_to_past: HashMap<String, String>,

    last_timings: Option<InferenceTimings>,
}

impl Chatterbox {
    pub fn load(paths: &hf::ChatterboxPaths) -> Result<Self> {
        Self::load_with(paths, &SessionConfig::default())
    }

    pub fn load_with(paths: &hf::ChatterboxPaths, cfg: &SessionConfig) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(&paths.tokenizer_json)
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        let conditional_decoder = session_builder(cfg)
            .context("conditional_decoder session builder")?
            .commit_from_file(&paths.conditional_decoder)
            .context("failed to load conditional_decoder.onnx")?;
        let speech_encoder = session_builder(cfg)
            .context("speech_encoder session builder")?
            .commit_from_file(&paths.speech_encoder)
            .context("failed to load speech_encoder.onnx")?;
        let embed_tokens = session_builder(cfg)
            .context("embed_tokens session builder")?
            .commit_from_file(&paths.embed_tokens)
            .context("failed to load embed_tokens.onnx")?;
        let language_model = session_builder(cfg)
            .context("language_model session builder")?
            .commit_from_file(&paths.language_model)
            .context("failed to load language_model.onnx")?;

        let speech_encoder_audio_input = speech_encoder
            .inputs()
            .iter()
            .find(|i| i.name() == "audio_values" || i.name() == "input_values")
            .map(|i| i.name().to_string())
            .ok_or_else(|| {
                anyhow!("speech_encoder missing expected audio input (audio_values/input_values)")
            })?;
        let speech_encoder_attention_mask_input = speech_encoder
            .inputs()
            .iter()
            .find(|i| i.name() == "attention_mask")
            .map(|i| i.name().to_string());

        let lm_inputs_embeds_type = language_model
            .inputs()
            .iter()
            .find(|i| i.name() == "inputs_embeds")
            .and_then(|i| elem_type_of_value_type(&i.dtype()))
            .unwrap_or(ElemType::F32);

        let (kv_type, kv_shape, past_kv_names, present_to_past) =
            inspect_kv_cache_io(&language_model)?;

        Ok(Self {
            tokenizer,
            conditional_decoder,
            speech_encoder,
            embed_tokens,
            language_model,
            speech_encoder_audio_input,
            speech_encoder_attention_mask_input,
            lm_inputs_embeds_type,
            kv_type,
            kv_shape,
            past_kv_names,
            present_to_past,
            last_timings: None,
        })
    }

    pub fn take_last_timings(&mut self) -> Option<InferenceTimings> {
        self.last_timings.take()
    }

    pub fn print_io(&self) {
        fn print_session(name: &str, s: &Session) {
            println!("== {name} ==");
            println!("inputs:");
            for i in s.inputs() {
                println!("  - {}: {:?}", i.name(), i.dtype());
            }
            println!("outputs:");
            for o in s.outputs() {
                println!("  - {}: {:?}", o.name(), o.dtype());
            }
        }
        print_session("embed_tokens", &self.embed_tokens);
        print_session("speech_encoder", &self.speech_encoder);
        print_session("language_model", &self.language_model);
        print_session("conditional_decoder", &self.conditional_decoder);
        println!(
            "lm_inputs_embeds_type={:?}, kv_type={:?}, kv_shape={:?}",
            self.lm_inputs_embeds_type, self.kv_type, self.kv_shape
        );
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        voice_wav: &Path,
        max_new_tokens: usize,
        repetition_penalty: f32,
    ) -> Result<Vec<f32>> {
        // 1) Load reference audio and run speech encoder.
        let audio_24k = audio::read_wav_as_f32_mono_24k(voice_wav)?;
        if audio_24k.is_empty() {
            bail!("voice wav has no samples");
        }

        let outputs = self.run_speech_encoder(&audio_24k)?;
        self.synthesize_inner(
            text,
            outputs.cond_emb,
            outputs.prompt_token,
            outputs.speaker_embeddings,
            outputs.speaker_features,
            max_new_tokens,
            repetition_penalty,
        )
    }

    pub fn encode_voice_profile(
        &mut self,
        voice_wav: &Path,
        repo_id: &str,
        revision: &str,
        dtype: &str,
    ) -> Result<VoiceProfile> {
        let audio_24k = audio::read_wav_as_f32_mono_24k(voice_wav)?;
        if audio_24k.is_empty() {
            bail!("voice wav has no samples");
        }

        let outputs = self.run_speech_encoder(&audio_24k)?;

        let (audio_features_shape, audio_features_data) =
            extract_tensor_f32(&outputs.cond_emb).context("extract audio_features")?;
        let (audio_tokens_shape, audio_tokens_data) =
            extract_tensor_i64(&outputs.prompt_token).context("extract audio_tokens")?;
        let (speaker_embeddings_shape, speaker_embeddings_data) =
            extract_tensor_f32(&outputs.speaker_embeddings)
                .context("extract speaker_embeddings")?;
        let (speaker_features_shape, speaker_features_data) =
            extract_tensor_f32(&outputs.speaker_features).context("extract speaker_features")?;

        Ok(VoiceProfile {
            format_version: 1,
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            dtype: dtype.to_string(),
            sample_rate_hz: TARGET_SAMPLE_RATE,
            audio_features: TensorF32 {
                shape: audio_features_shape,
                data: audio_features_data,
            },
            audio_tokens: TensorI64 {
                shape: audio_tokens_shape,
                data: audio_tokens_data,
            },
            speaker_embeddings: TensorF32 {
                shape: speaker_embeddings_shape,
                data: speaker_embeddings_data,
            },
            speaker_features: TensorF32 {
                shape: speaker_features_shape,
                data: speaker_features_data,
            },
        })
    }

    pub fn synthesize_with_voice_profile(
        &mut self,
        text: &str,
        repo_id: &str,
        revision: &str,
        dtype: &str,
        profile: &VoiceProfile,
        max_new_tokens: usize,
        repetition_penalty: f32,
    ) -> Result<Vec<f32>> {
        if profile.format_version != 1 {
            bail!(
                "unsupported voice profile format_version {}",
                profile.format_version
            );
        }
        if profile.sample_rate_hz != TARGET_SAMPLE_RATE {
            bail!(
                "voice profile sample_rate_hz {} does not match model sample rate {}",
                profile.sample_rate_hz,
                TARGET_SAMPLE_RATE
            );
        }
        if profile.repo_id != repo_id || profile.revision != revision || profile.dtype != dtype {
            bail!(
                "voice profile was created for {}/{} dtype={} but you requested {}/{} dtype={}",
                profile.repo_id,
                profile.revision,
                profile.dtype,
                repo_id,
                revision,
                dtype
            );
        }

        let cond_emb: DynValue = Tensor::from_array((
            profile.audio_features.shape.clone(),
            profile.audio_features.data.clone(),
        ))
        .context("failed to build audio_features tensor")?
        .into_dyn();

        let prompt_token: DynValue = Tensor::from_array((
            profile.audio_tokens.shape.clone(),
            profile.audio_tokens.data.clone(),
        ))
        .context("failed to build audio_tokens tensor")?
        .into_dyn();

        let speaker_embeddings = tensor_from_f32_profile(
            &profile.speaker_embeddings,
            elem_type_of_named_input(&self.conditional_decoder, "speaker_embeddings")
                .unwrap_or(ElemType::F32),
        )
        .context("failed to build speaker_embeddings tensor")?;

        let speaker_features = tensor_from_f32_profile(
            &profile.speaker_features,
            elem_type_of_named_input(&self.conditional_decoder, "speaker_features")
                .unwrap_or(ElemType::F32),
        )
        .context("failed to build speaker_features tensor")?;

        self.synthesize_inner(
            text,
            cond_emb,
            prompt_token,
            speaker_embeddings,
            speaker_features,
            max_new_tokens,
            repetition_penalty,
        )
    }

    fn synthesize_inner(
        &mut self,
        text: &str,
        cond_emb: DynValue,
        prompt_token: DynValue,
        speaker_embeddings: DynValue,
        speaker_features: DynValue,
        max_new_tokens: usize,
        repetition_penalty: f32,
    ) -> Result<Vec<f32>> {
        let t_total = Instant::now();

        // 1) Tokenize text.
        let t_tok = Instant::now();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?;
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        if input_ids.is_empty() {
            bail!("tokenizer produced zero tokens");
        }
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;

        // 2) Autoregressive loop over language model.
        let mut embed_ms = 0.0f64;
        let mut lm_ms = 0.0f64;
        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut stop_token: Option<i64> = None;

        let mut attention_mask: Vec<i64> = Vec::new();
        let mut position_ids: Vec<i64> = Vec::new();
        let mut past_kv: HashMap<String, DynValue> = self.init_past_kv()?;

        for step in 0..max_new_tokens {
            let (inputs_embeds, seq_len_for_mask) = if step == 0 {
                let t = Instant::now();
                let embeds = self.run_embed_tokens(&input_ids)?;
                embed_ms += t.elapsed().as_secs_f64() * 1000.0;
                let merged = concat_inputs_embeds(&cond_emb, &embeds, self.lm_inputs_embeds_type)?;
                let seq_len = tensor_seq_len(&merged)?;
                (merged, seq_len)
            } else {
                let t = Instant::now();
                let embeds = self.run_embed_tokens(&input_ids)?;
                embed_ms += t.elapsed().as_secs_f64() * 1000.0;
                let seq_len = tensor_seq_len(&embeds)?;
                (embeds, seq_len)
            };

            if step == 0 {
                attention_mask = vec![1; seq_len_for_mask];
                position_ids = (0..seq_len_for_mask as i64).collect();
            } else {
                attention_mask.push(1);
                let next_pos = position_ids.last().copied().unwrap_or(0) + 1;
                position_ids = vec![next_pos];
            }

            let attention_mask_tensor: Tensor<i64> =
                Tensor::from_array(([1usize, attention_mask.len()], attention_mask.clone()))
                    .context("failed to build attention_mask tensor")?;
            let position_ids_tensor: Tensor<i64> =
                Tensor::from_array(([1usize, position_ids.len()], position_ids.clone()))
                    .context("failed to build position_ids tensor")?;

            let mut feed: HashMap<String, SessionInputValue> = HashMap::new();
            feed.insert("inputs_embeds".to_string(), inputs_embeds.into());
            feed.insert("attention_mask".to_string(), attention_mask_tensor.into());
            feed.insert("position_ids".to_string(), position_ids_tensor.into());

            for (k, v) in past_kv.iter() {
                feed.insert(k.clone(), clone_dyn_value(v)?.into());
            }

            let t = Instant::now();
            let mut outputs = self
                .language_model
                .run(SessionInputs::from(feed))
                .context("language_model inference failed")?;
            lm_ms += t.elapsed().as_secs_f64() * 1000.0;

            let logits_value = outputs
                .get("logits")
                .or_else(|| Some(&outputs[0]))
                .ok_or_else(|| anyhow!("language_model produced no outputs"))?;

            let logits_last =
                extract_last_logits_f32(logits_value).context("failed to extract logits")?;
            let mut adjusted = logits_last;
            let vocab_size = adjusted.len() as i64;
            if stop_token.is_none() {
                stop_token = Some(vocab_size - 1);
            }
            if repetition_penalty != 1.0 {
                apply_repetition_penalty(&mut adjusted, &generated_tokens, repetition_penalty);
            }
            let next_token = argmax_i64(&adjusted)?;

            generated_tokens.push(next_token);

            if Some(next_token) == stop_token {
                break;
            }

            // Next iteration: embed the newly generated token.
            input_ids = vec![next_token];

            // Update past_key_values (present -> past).
            let mut new_past: HashMap<String, DynValue> = HashMap::new();
            for (present_name, past_name) in &self.present_to_past {
                if let Some(v) = outputs.remove(present_name.as_str()) {
                    new_past.insert(past_name.clone(), v);
                }
            }
            if !new_past.is_empty() {
                past_kv = new_past;
            }
        }

        // The language model's vocabulary is larger than the decoder's. In this HF export, `logits` has a vocab size
        // of 6563 (so STOP is typically 6562), while the decoder's embedding table only supports ids 0..=6560.
        //
        // We:
        // - use STOP to terminate generation
        // - drop STOP (and any other out-of-range ids) before decoding
        generated_tokens.extend([SILENCE_TOKEN, SILENCE_TOKEN, SILENCE_TOKEN]);

        let (prompt_shape, prompt_data) =
            extract_tensor_i64(&prompt_token).context("extract prompt_token (audio_tokens)")?;
        if prompt_shape.len() != 2 || prompt_shape[0] != 1 {
            bail!("unexpected prompt_token shape: {:?}", prompt_shape);
        }
        let decoder_vocab_size = stop_token.unwrap_or(6562) - 1;
        if SILENCE_TOKEN >= decoder_vocab_size {
            bail!(
                "silence token id {} is out of range for decoder_vocab_size {}",
                SILENCE_TOKEN,
                decoder_vocab_size
            );
        }
        let mut decoder_tokens = prompt_data;
        decoder_tokens.extend(
            generated_tokens
                .into_iter()
                .filter(|t| *t < decoder_vocab_size),
        );

        // 4) Conditional decoder -> waveform.
        let t_dec = Instant::now();
        let out =
            self.run_conditional_decoder(&decoder_tokens, &speaker_embeddings, &speaker_features)?;
        let decoder_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

        self.last_timings = Some(InferenceTimings {
            tokenize_ms: tok_ms,
            embed_ms,
            lm_ms,
            decoder_ms,
            total_ms: t_total.elapsed().as_secs_f64() * 1000.0,
        });

        Ok(out)
    }

    fn run_embed_tokens(&mut self, input_ids: &[i64]) -> Result<DynValue> {
        let input_tensor: Tensor<i64> =
            Tensor::from_array(([1usize, input_ids.len()], input_ids.to_vec()))
                .context("input_ids tensor")?;
        let mut feed: HashMap<String, SessionInputValue> = HashMap::new();
        feed.insert("input_ids".to_string(), input_tensor.into());
        let outputs = self
            .embed_tokens
            .run(SessionInputs::from(feed))
            .context("embed_tokens inference failed")?;
        let v = &outputs[0];

        // Return as a new tensor so it can be retyped/owned independently.
        retype_tensor(v, self.lm_inputs_embeds_type)
    }

    fn run_speech_encoder(&mut self, input_values: &[f32]) -> Result<SpeechEncoderOutputs> {
        let input_values_tensor: Tensor<f32> =
            Tensor::from_array(([1usize, input_values.len()], input_values.to_vec()))
                .context("input_values tensor")?;

        let mut feed: HashMap<String, SessionInputValue> = HashMap::new();
        feed.insert(
            self.speech_encoder_audio_input.clone(),
            input_values_tensor.into(),
        );
        if let Some(mask_name) = &self.speech_encoder_attention_mask_input {
            let attention_mask = vec![1i64; input_values.len()];
            let attention_mask_tensor: Tensor<i64> =
                Tensor::from_array(([1usize, attention_mask.len()], attention_mask))
                    .context("attention_mask tensor")?;
            feed.insert(mask_name.clone(), attention_mask_tensor.into());
        }

        let outputs = self
            .speech_encoder
            .run(SessionInputs::from(feed))
            .context("speech_encoder inference failed")?;

        let mut outputs = outputs;
        // Names shown by `inspect` for this repo:
        // audio_features, audio_tokens, speaker_embeddings, speaker_features
        let cond_emb = outputs
            .remove("audio_features")
            .or_else(|| outputs.remove("cond_emb"))
            .ok_or_else(|| anyhow!("speech_encoder missing audio_features/cond_emb output"))?;
        let prompt_token = outputs
            .remove("audio_tokens")
            .or_else(|| outputs.remove("prompt_token"))
            .ok_or_else(|| anyhow!("speech_encoder missing audio_tokens/prompt_token output"))?;
        let speaker_embeddings = outputs
            .remove("speaker_embeddings")
            .ok_or_else(|| anyhow!("speech_encoder missing speaker_embeddings output"))?;
        let speaker_features = outputs
            .remove("speaker_features")
            .ok_or_else(|| anyhow!("speech_encoder missing speaker_features output"))?;

        Ok(SpeechEncoderOutputs {
            cond_emb,
            prompt_token,
            speaker_embeddings,
            speaker_features,
        })
    }

    fn run_conditional_decoder(
        &mut self,
        speech_tokens: &[i64],
        speaker_embeddings: &DynValue,
        speaker_features: &DynValue,
    ) -> Result<Vec<f32>> {
        let speech_tokens_tensor: Tensor<i64> =
            Tensor::from_array(([1usize, speech_tokens.len()], speech_tokens.to_vec()))
                .context("speech_tokens tensor")?;

        let mut feed: HashMap<String, SessionInputValue> = HashMap::new();
        feed.insert("speech_tokens".to_string(), speech_tokens_tensor.into());
        feed.insert(
            "speaker_embeddings".to_string(),
            clone_dyn_value(speaker_embeddings)?.into(),
        );
        feed.insert(
            "speaker_features".to_string(),
            clone_dyn_value(speaker_features)?.into(),
        );

        let outputs = self
            .conditional_decoder
            .run(SessionInputs::from(feed))
            .context("conditional_decoder inference failed")?;

        let wav = &outputs[0];
        let (shape, data) = extract_tensor_f32(wav).context("extract wav tensor")?;
        // Expect [1, N] or [N].
        let samples = match shape.as_slice() {
            [1, _n] => data,
            [_n] => data,
            _ => bail!("unexpected wav output shape: {:?}", shape),
        };
        Ok(samples)
    }

    fn init_past_kv(&self) -> Result<HashMap<String, DynValue>> {
        let (b, h, _t, d) = self.kv_shape;
        let mut out = HashMap::new();

        for name in &self.past_kv_names {
            let v = match self.kv_type {
                ElemType::F16 => {
                    let allocator = ort::memory::Allocator::default();
                    let t = ort::value::DynTensor::new(
                        &allocator,
                        TensorElementType::Float16,
                        [b, h, 0usize, d],
                    )?;
                    t.into_dyn()
                }
                ElemType::F32 => {
                    let allocator = ort::memory::Allocator::default();
                    let t = ort::value::DynTensor::new(
                        &allocator,
                        TensorElementType::Float32,
                        [b, h, 0usize, d],
                    )?;
                    t.into_dyn()
                }
            };
            out.insert(name.clone(), v);
        }
        Ok(out)
    }
}

fn session_builder(cfg: &SessionConfig) -> Result<ort::session::builder::SessionBuilder> {
    let mut b = Session::builder().context("ORT Session builder")?;
    if let Some(n) = cfg.intra_threads {
        b = b.with_intra_threads(n)?;
    }
    if cfg.parallel_execution {
        b = b.with_parallel_execution(true)?;
        if let Some(n) = cfg.inter_threads {
            b = b.with_inter_threads(n)?;
        }
    }

    // Execution providers:
    // - When `Auto`, prefer platform accelerators when built + available; otherwise fall back to CPU.
    // - When a specific EP is requested, attempt it and error if not built.
    let requested = cfg.execution_provider;

    fn cpu_fallback() -> ort::execution_providers::ExecutionProviderDispatch {
        ort::execution_providers::CPUExecutionProvider::default().build()
    }

    #[cfg(feature = "coreml")]
    fn coreml_ep(cfg: &SessionConfig) -> ort::execution_providers::ExecutionProviderDispatch {
        use ort::ep::coreml::{ComputeUnits, CoreML, ModelFormat, SpecializationStrategy};
        let mut coreml = CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_specialization_strategy(SpecializationStrategy::FastPrediction)
            .with_compute_units(ComputeUnits::CPUAndNeuralEngine);
        if let Some(dir) = &cfg.coreml_cache_dir {
            coreml = coreml.with_model_cache_dir(dir.display().to_string());
        }
        coreml.build()
    }

    #[cfg(feature = "directml")]
    fn directml_ep() -> ort::execution_providers::ExecutionProviderDispatch {
        ort::execution_providers::DirectMLExecutionProvider::default().build()
    }

    #[cfg(feature = "cuda")]
    fn cuda_ep() -> ort::execution_providers::ExecutionProviderDispatch {
        ort::execution_providers::CUDAExecutionProvider::default().build()
    }

    let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();

    match requested {
        ExecutionProvider::Cpu => {
            eps.push(cpu_fallback());
            Ok(b.with_execution_providers(eps)?)
        }
        ExecutionProvider::CoreML => {
            #[cfg(feature = "coreml")]
            {
                use ort::ep::ExecutionProvider as _;
                if !ort::ep::coreml::CoreML::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("CoreML EP is not available in this ONNX Runtime build");
                }
                eps.push(coreml_ep(cfg));
                eps.push(cpu_fallback());
                Ok(b.with_execution_providers(eps)?)
            }
            #[cfg(not(feature = "coreml"))]
            {
                bail!(
                    "CoreML execution provider requested, but cbx was built without the `coreml` feature"
                );
            }
        }
        ExecutionProvider::DirectML => {
            #[cfg(feature = "directml")]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if !ort::execution_providers::DirectMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("DirectML EP is not available in this ONNX Runtime build");
                }
                eps.push(directml_ep());
                eps.push(cpu_fallback());
                Ok(b.with_execution_providers(eps)?)
            }
            #[cfg(not(feature = "directml"))]
            {
                bail!(
                    "DirectML execution provider requested, but cbx was built without the `directml` feature"
                );
            }
        }
        ExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if !ort::execution_providers::CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("CUDA EP is not available in this ONNX Runtime build");
                }
                eps.push(cuda_ep());
                eps.push(cpu_fallback());
                Ok(b.with_execution_providers(eps)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "CUDA execution provider requested, but cbx was built without the `cuda` feature"
                );
            }
        }
        ExecutionProvider::Auto => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                use ort::ep::ExecutionProvider as _;
                if ort::ep::coreml::CoreML::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    eps.push(coreml_ep(cfg));
                }
            }
            #[cfg(all(feature = "cuda", target_os = "windows"))]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if ort::execution_providers::CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    eps.push(cuda_ep());
                }
            }
            #[cfg(all(feature = "directml", target_os = "windows"))]
            {
                use ort::execution_providers::ExecutionProvider as _;
                if ort::execution_providers::DirectMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    eps.push(directml_ep());
                }
            }

            eps.push(cpu_fallback());
            Ok(b.with_execution_providers(eps)?)
        }
    }
}

#[derive(Debug)]
struct SpeechEncoderOutputs {
    cond_emb: DynValue,
    prompt_token: DynValue,
    speaker_embeddings: DynValue,
    speaker_features: DynValue,
}

fn inspect_kv_cache_io(
    language_model: &Session,
) -> Result<(
    ElemType,
    (usize, usize, usize, usize),
    Vec<String>,
    HashMap<String, String>,
)> {
    let mut past_kv_names = Vec::new();
    let mut kv_type = None;
    let mut kv_heads = None;
    let mut kv_head_dim = None;

    for i in language_model.inputs() {
        if !i.name().contains("past_key_values") {
            continue;
        }
        past_kv_names.push(i.name().to_string());
        kv_type = kv_type.or_else(|| elem_type_of_value_type(&i.dtype()));

        if let ValueType::Tensor { shape, .. } = &i.dtype() {
            if shape.len() == 4 {
                if shape[1] > 0 {
                    kv_heads = Some(shape[1] as usize);
                }
                if shape[3] > 0 {
                    kv_head_dim = Some(shape[3] as usize);
                }
            }
        }
    }

    let kv_type = kv_type.unwrap_or(ElemType::F32);
    let h = kv_heads.unwrap_or(DEFAULT_NUM_KV_HEADS);
    let d = kv_head_dim.unwrap_or(DEFAULT_HEAD_DIM);
    let kv_shape = (1usize, h, 0usize, d);

    // Map present -> past (stable even if indexing order changes).
    let mut present_to_past = HashMap::new();
    for o in language_model.outputs() {
        let oname = o.name();
        if oname.starts_with("present.") {
            let past = oname.replacen("present.", "past_key_values.", 1);
            if past_kv_names.iter().any(|n| n == &past) {
                present_to_past.insert(oname.to_string(), past);
            }
            continue;
        }
        if !oname.contains("present_key_values") {
            continue;
        }
        let past = oname.replace("present_key_values", "past_key_values");
        if past_kv_names.iter().any(|n| n == &past) {
            present_to_past.insert(oname.to_string(), past);
        }
    }

    Ok((kv_type, kv_shape, past_kv_names, present_to_past))
}

fn tensor_seq_len(t: &DynValue) -> Result<usize> {
    if let Ok((shape, _)) = t.try_extract_tensor::<f32>() {
        if shape.len() < 2 {
            bail!("inputs_embeds has rank < 2: {:?}", shape);
        }
        return Ok(shape[1] as usize);
    }
    if let Ok((shape, _)) = t.try_extract_tensor::<f16>() {
        if shape.len() < 2 {
            bail!("inputs_embeds has rank < 2: {:?}", shape);
        }
        return Ok(shape[1] as usize);
    }
    bail!("inputs_embeds has unsupported dtype");
}

fn concat_inputs_embeds(a: &DynValue, b: &DynValue, out_type: ElemType) -> Result<DynValue> {
    let (a_shape, a_data) = extract_tensor_f32(a).context("extract cond_emb")?;
    let (b_shape, b_data) = extract_tensor_f32(b).context("extract embeds")?;

    if a_shape.len() != 3 || b_shape.len() != 3 {
        bail!(
            "expected 3D tensors for embeddings, got {:?} and {:?}",
            a_shape,
            b_shape
        );
    }
    if a_shape[0] != 1 || b_shape[0] != 1 || a_shape[2] != b_shape[2] {
        bail!(
            "embedding shapes incompatible: {:?} and {:?}",
            a_shape,
            b_shape
        );
    }
    let a_seq = a_shape[1];
    let b_seq = b_shape[1];
    let hidden = a_shape[2];
    let mut out = Vec::with_capacity((a_seq + b_seq) * hidden);

    out.extend_from_slice(&a_data);
    out.extend_from_slice(&b_data);

    match out_type {
        ElemType::F32 => Ok(
            Tensor::from_array((vec![1usize, a_seq + b_seq, hidden], out))
                .context("create concat f32 tensor")?
                .into_dyn(),
        ),
        ElemType::F16 => {
            let out16: Vec<f16> = out.into_iter().map(f16::from_f32).collect();
            Ok(
                Tensor::from_array((vec![1usize, a_seq + b_seq, hidden], out16))
                    .context("create concat f16 tensor")?
                    .into_dyn(),
            )
        }
    }
}

fn retype_tensor(value: &DynValue, out_type: ElemType) -> Result<DynValue> {
    match out_type {
        ElemType::F32 => {
            let (shape, data) = extract_tensor_f32(value)?;
            Ok(Tensor::from_array((shape, data))
                .context("retype to f32")?
                .into_dyn())
        }
        ElemType::F16 => {
            let (shape, data) = extract_tensor_f32(value)?;
            let data16: Vec<f16> = data.into_iter().map(f16::from_f32).collect();
            Ok(Tensor::from_array((shape, data16))
                .context("retype to f16")?
                .into_dyn())
        }
    }
}

fn extract_tensor_f32(value: &DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        let shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
        return Ok((shape, data.to_vec()));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
        let shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
        let data = data.iter().map(|&v| v.to_f32()).collect::<Vec<_>>();
        return Ok((shape, data));
    }
    bail!("unsupported tensor dtype (expected f16 or f32)")
}

fn extract_tensor_i64(value: &DynValue) -> Result<(Vec<usize>, Vec<i64>)> {
    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        let shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
        return Ok((shape, data.to_vec()));
    }
    bail!("unsupported tensor dtype (expected i64)")
}

fn extract_last_logits_f32(value: &DynValue) -> Result<Vec<f32>> {
    let (shape, data) = extract_tensor_f32(value)?;
    match shape.as_slice() {
        [1, seq, vocab] => {
            if *seq == 0 {
                bail!("logits seq_len=0");
            }
            let offset = (seq - 1) * vocab;
            Ok(data[offset..offset + vocab].to_vec())
        }
        [1, _vocab] => Ok(data),
        _ => bail!("unexpected logits shape: {:?}", shape),
    }
}

fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[i64], penalty: f32) {
    if penalty <= 0.0 {
        return;
    }
    for &t in generated_tokens {
        let idx = t as usize;
        if idx >= logits.len() {
            continue;
        }
        let l = logits[idx];
        logits[idx] = if l < 0.0 { l * penalty } else { l / penalty };
    }
}

fn argmax_i64(logits: &[f32]) -> Result<i64> {
    if logits.is_empty() {
        bail!("empty logits");
    }
    let mut best_i = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(best_i as i64)
}

fn _assert_sample_rate() {
    let _ = TARGET_SAMPLE_RATE;
}

fn clone_dyn_value(value: &DynValue) -> Result<DynValue> {
    value
        .view()
        .try_upgrade()
        .map_err(|_| anyhow!("failed to upgrade DynValue to owned value"))
        .map(|v| v.into_dyn())
}

#[derive(Debug, Clone, Default)]
pub struct InferenceTimings {
    pub tokenize_ms: f64,
    pub embed_ms: f64,
    pub lm_ms: f64,
    pub decoder_ms: f64,
    pub total_ms: f64,
}
