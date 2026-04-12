//! Optimized streaming pipeline for overlapped audio processing.
//!
//! This module implements a high-performance streaming pipeline that processes
//! audio through VAD → STT with overlapping execution. The key optimization
//! is processing multiple STT chunks in parallel while maintaining VAD state.
//!
//! # Architecture
//!
//! ```text
//! Sequential (Current):
//! Audio → VAD → STT → Result (wait for each chunk)
//!
//! Overlapped (Optimized):
//! Audio → VAD → ┬→ STT (chunk 1) ──→ Result
//!               ├→ STT (chunk 2) ──→ Result
//!               └→ STT (chunk 3) ──→ Result
//! ```
//!
//! # Key Techniques
//!
//! - **Parallel STT processing**: Multiple chunks transcribed simultaneously
//! - **Bounded concurrency**: Limit active STT tasks for memory control
//! - **Chunk-based streaming**: Split audio into overlapping windows
//! - **Backpressure handling**: Tokio channels prevent memory overflow
//!
//! # Performance
//!
//! Target 50% latency reduction through parallelism:
//! - Sequential: 2000ms STT × 3 chunks = 6000ms total
//! - Parallel (3x): 2000ms STT (all chunks) = 2000ms total

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, mpsc};

use crate::error::VoxError;
use crate::traits::{SttBackend, VadEvent};
use crate::types::{SttResult, Utterance};

/// Configuration for the streaming pipeline.
#[derive(Debug, Clone)]
pub struct StreamingPipelineConfig {
    /// Maximum number of STT tasks running in parallel.
    pub max_parallel_stt: usize,
    /// Channel buffer size for backpressure control.
    pub channel_buffer_size: usize,
    /// Enable detailed per-chunk metrics logging.
    pub enable_detailed_metrics: bool,
}

impl Default for StreamingPipelineConfig {
    fn default() -> Self {
        Self {
            max_parallel_stt: 3,     // 3 STT tasks in parallel
            channel_buffer_size: 16, // Buffer up to 16 chunks
            enable_detailed_metrics: false,
        }
    }
}

/// Statistics for a single chunk's processing.
#[derive(Debug, Clone)]
pub struct ChunkStats {
    pub chunk_id: u64,
    pub vad_latency_ms: u64,
    pub stt_latency_ms: u64,
    pub total_latency_ms: u64,
}

/// Pipeline performance metrics.
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    pub total_chunks: u64,
    pub avg_stt_latency_ms: f64,
    pub avg_total_latency_ms: f64,
    pub chunks_dropped: u64,
    pub peak_parallel_stt: usize,
}

impl PipelineMetrics {
    /// Update metrics with a new chunk's statistics.
    pub fn update(&mut self, stats: ChunkStats) {
        self.total_chunks += 1;
        let n = self.total_chunks as f64;

        self.avg_stt_latency_ms =
            self.avg_stt_latency_ms * ((n - 1.0) / n) + stats.stt_latency_ms as f64 / n;

        self.avg_total_latency_ms =
            self.avg_total_latency_ms * ((n - 1.0) / n) + stats.total_latency_ms as f64 / n;
    }
}

/// A detected utterance ready for STT processing.
#[derive(Debug, Clone)]
struct SttWork {
    utterance: Utterance,
    chunk_id: u64,
    timestamp: Instant,
}

/// The streaming pipeline orchestrator.
pub struct StreamingPipeline {
    config: StreamingPipelineConfig,
    stt: Arc<dyn SttBackend>,
    metrics: Arc<Mutex<PipelineMetrics>>,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline.
    pub fn new(config: StreamingPipelineConfig, stt: Arc<dyn SttBackend>) -> Self {
        Self {
            config,
            stt,
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }

    /// Get current pipeline metrics.
    pub async fn metrics(&self) -> PipelineMetrics {
        self.metrics.lock().await.clone()
    }

    /// Process VAD events through parallel STT pipeline.
    ///
    /// Takes VAD events from the input channel and processes speech utterances
    /// through STT in parallel, up to `max_parallel_stt` concurrent tasks.
    pub async fn process_vad_stream(
        &self,
        mut vad_rx: mpsc::Receiver<VadEvent>,
    ) -> Result<mpsc::Receiver<SttResult>, VoxError> {
        let (result_tx, result_rx) = mpsc::channel(self.config.channel_buffer_size);

        let stt = self.stt.clone();
        let max_parallel = self.config.max_parallel_stt;
        let metrics = self.metrics.clone();
        let enable_metrics = self.config.enable_detailed_metrics;

        tokio::spawn(async move {
            let mut active_tasks: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
            let mut chunk_id = 0u64;

            loop {
                tokio::select! {
                    // Receive new VAD events if we have capacity
                    event_opt = vad_rx.recv(), if active_tasks.len() < max_parallel => {
                        match event_opt {
                            Some(event) => {
                        if let VadEvent::SpeechEnd(utterance) = event {
                            let stt_clone = stt.clone();
                            let result_tx_clone = result_tx.clone();
                            let metrics_clone = metrics.clone();
                            let work = SttWork {
                                utterance,
                                chunk_id,
                                timestamp: Instant::now(),
                            };
                            chunk_id += 1;

                            // Track peak parallelism
                            {
                                let mut m = metrics_clone.lock().await;
                                m.peak_parallel_stt = m.peak_parallel_stt.max(active_tasks.len() + 1);
                            }

                            active_tasks.spawn(async move {
                                let t_start = Instant::now();

                                match stt_clone.transcribe(&work.utterance).await {
                                    Ok(result) => {
                                        let stt_latency = t_start.elapsed().as_millis() as u64;
                                        let total_latency = work.timestamp.elapsed().as_millis() as u64;

                                        if enable_metrics {
                                            let stats = ChunkStats {
                                                chunk_id: work.chunk_id,
                                                vad_latency_ms: 0,
                                                stt_latency_ms: stt_latency,
                                                total_latency_ms: total_latency,
                                            };
                                            metrics_clone.lock().await.update(stats);
                                        }

                                        if !result.text.is_empty()
                                            && result_tx_clone.send(result).await.is_err()
                                        {
                                            tracing::warn!("STT pipeline: downstream closed");
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(chunk_id = work.chunk_id, error = %e, "STT failed");
                                        metrics_clone.lock().await.chunks_dropped += 1;
                                    }
                                }
                            });
                        }
                            }
                            None => break, // VAD stream closed
                        }
                    }

                    // Process completed tasks
                    Some(_) = active_tasks.join_next() => {
                        // Task completed, slot freed for next chunk
                    }

                    // Exit when input closed and all tasks done
                    else => {
                        break;
                    }
                }
            }

            // Wait for remaining tasks
            while active_tasks.join_next().await.is_some() {}

            let final_metrics = metrics.lock().await.clone();
            tracing::debug!(
                metrics = ?final_metrics,
                "streaming STT pipeline completed"
            );
        });

        Ok(result_rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = StreamingPipelineConfig::default();
        assert_eq!(config.max_parallel_stt, 3);
        assert_eq!(config.channel_buffer_size, 16);
        assert!(!config.enable_detailed_metrics);
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let mut metrics = PipelineMetrics::default();

        let stats1 = ChunkStats {
            chunk_id: 0,
            vad_latency_ms: 50,
            stt_latency_ms: 2000,
            total_latency_ms: 2050,
        };

        metrics.update(stats1);
        assert_eq!(metrics.total_chunks, 1);
        assert_eq!(metrics.avg_stt_latency_ms, 2000.0);

        let stats2 = ChunkStats {
            chunk_id: 1,
            vad_latency_ms: 50,
            stt_latency_ms: 1500,
            total_latency_ms: 1550,
        };

        metrics.update(stats2);
        assert_eq!(metrics.total_chunks, 2);
        assert_eq!(metrics.avg_stt_latency_ms, 1750.0); // (2000 + 1500) / 2
    }
}
