//! Neural network models for Qwen3-TTS
//!
//! This module contains:
//! - `transformer`: Shared building blocks (KVCache, RoPE, RoPEType, Attention, MLP, DecoderLayer)
//! - `talker`: TalkerModel for semantic token generation
//! - `code_predictor`: Acoustic token predictor
//! - `speaker`: Speaker encoder (ECAPA-TDNN)
//! - `codec`: Audio codec for encoding/decoding
//! - `config`: Model configuration

pub mod code_predictor;
pub mod codec;
pub mod config;
pub mod fused_ops;
pub mod kv_cache;
#[cfg(feature = "quantized")]
pub mod quantized;
pub mod speaker;
pub mod talker;
pub mod transformer;

pub use code_predictor::{CodePredictor, CodePredictorConfig};
pub use config::{ModelType, ParsedModelConfig, Qwen3TTSConfig, SpeakerEncoderConfig};
#[cfg(feature = "quantized")]
pub use config::{QuantizationConfig, QuantizationDtype, QuantizationLayers};
pub use kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};
#[cfg(feature = "quantized")]
pub use quantized::{quantize_linear_layer, quantize_model_layers, QuantizedLinear};
pub use talker::{TalkerConfig, TalkerModel};
pub use transformer::{MRoPE, RoPEType, RotaryEmbedding};
