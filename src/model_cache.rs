//! Model caching to eliminate cold starts.
//!
//! Provides an optional LRU cache for STT and TTS backends.
//! - CLI mode: always caches (better UX)
//! - Server mode: opt-in via --cache-models flag (backwards compat)
//!
//! Thread-safe singleton pattern using `Arc<Mutex<HashMap>>`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::VoxError;
use crate::traits::{SttBackend, TtsBackend};

/// Cache key uniquely identifying a model.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct CacheKey {
    /// Model type (stt, tts, vad)
    pub kind: String,
    /// Model path or identifier
    pub identifier: String,
}

impl CacheKey {
    pub fn stt(path: &std::path::Path) -> Self {
        Self {
            kind: "stt".to_string(),
            identifier: path.display().to_string(),
        }
    }

    pub fn tts(path: &std::path::Path) -> Self {
        Self {
            kind: "tts".to_string(),
            identifier: path.display().to_string(),
        }
    }
}

/// Cached model entry with access time for LRU eviction.
struct CacheEntry {
    stt: Option<Arc<dyn SttBackend>>,
    tts: Option<Arc<dyn TtsBackend>>,
    last_access: Instant,
}

/// Model cache with LRU eviction.
///
/// # Example
///
/// ```ignore
/// let cache = ModelCache::new(3);
/// let stt = cache.get_or_load_stt(&path, || WhisperBackend::from_model(&path))?;
/// ```
pub struct ModelCache {
    inner: Arc<Mutex<CacheInner>>,
}

struct CacheInner {
    entries: HashMap<CacheKey, CacheEntry>,
    max_entries: usize,
    hits: u64,
    misses: u64,
}

impl ModelCache {
    /// Create a new model cache with LRU eviction.
    ///
    /// # Arguments
    /// - `max_entries`: Maximum number of models to cache. When exceeded,
    ///   the least recently used model is evicted.
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CacheInner {
                entries: HashMap::new(),
                max_entries,
                hits: 0,
                misses: 0,
            })),
        }
    }

    /// Get or load an STT backend.
    ///
    /// If the model is cached, returns the cached instance.
    /// Otherwise, calls `loader` to create a new instance and caches it.
    pub fn get_or_load_stt<F>(
        &self,
        key: CacheKey,
        loader: F,
    ) -> Result<Arc<dyn SttBackend>, VoxError>
    where
        F: FnOnce() -> Result<Arc<dyn SttBackend>, VoxError>,
    {
        let mut inner = self.inner.lock().unwrap();

        // Check cache - extract backend if present
        let cached_backend = if let Some(entry) = inner.entries.get_mut(&key) {
            if let Some(stt) = &entry.stt {
                entry.last_access = Instant::now();
                Some(Arc::clone(stt))
            } else {
                None
            }
        } else {
            None
        };

        // If cached, increment hits and return
        if let Some(backend) = cached_backend {
            inner.hits += 1;
            return Ok(backend);
        }

        // Cache miss - load model
        inner.misses += 1;
        drop(inner); // Release lock while loading

        let backend = loader()?;

        // Reacquire lock and insert
        let mut inner = self.inner.lock().unwrap();
        self.evict_lru_if_needed(&mut inner);

        inner.entries.insert(
            key,
            CacheEntry {
                stt: Some(Arc::clone(&backend)),
                tts: None,
                last_access: Instant::now(),
            },
        );

        Ok(backend)
    }

    /// Get or load a TTS backend.
    ///
    /// If the model is cached, returns the cached instance.
    /// Otherwise, calls `loader` to create a new instance and caches it.
    pub fn get_or_load_tts<F>(
        &self,
        key: CacheKey,
        loader: F,
    ) -> Result<Arc<dyn TtsBackend>, VoxError>
    where
        F: FnOnce() -> Result<Arc<dyn TtsBackend>, VoxError>,
    {
        let mut inner = self.inner.lock().unwrap();

        // Check cache - extract backend if present
        let cached_backend = if let Some(entry) = inner.entries.get_mut(&key) {
            if let Some(tts) = &entry.tts {
                entry.last_access = Instant::now();
                Some(Arc::clone(tts))
            } else {
                None
            }
        } else {
            None
        };

        // If cached, increment hits and return
        if let Some(backend) = cached_backend {
            inner.hits += 1;
            return Ok(backend);
        }

        // Cache miss - load model
        inner.misses += 1;
        drop(inner); // Release lock while loading

        let backend = loader()?;

        // Reacquire lock and insert
        let mut inner = self.inner.lock().unwrap();
        self.evict_lru_if_needed(&mut inner);

        inner.entries.insert(
            key,
            CacheEntry {
                stt: None,
                tts: Some(Arc::clone(&backend)),
                last_access: Instant::now(),
            },
        );

        Ok(backend)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().unwrap();
        CacheStats {
            entries: inner.entries.len(),
            max_entries: inner.max_entries,
            hits: inner.hits,
            misses: inner.misses,
            hit_rate: if inner.hits + inner.misses > 0 {
                (inner.hits as f64) / ((inner.hits + inner.misses) as f64)
            } else {
                0.0
            },
        }
    }

    /// Clear all cached models.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.hits = 0;
        inner.misses = 0;
    }

    /// Evict least recently used entry if cache is full.
    fn evict_lru_if_needed(&self, inner: &mut CacheInner) {
        if inner.entries.len() >= inner.max_entries {
            // Find LRU entry
            if let Some((lru_key, _)) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
            {
                let lru_key = lru_key.clone();
                inner.entries.remove(&lru_key);
                tracing::debug!(key = ?lru_key, "evicted LRU model from cache");
            }
        }
    }
}

impl Clone for ModelCache {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize))]
pub struct CacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Hit rate (hits / total requests)
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SttResult, TtsOutput, TtsRequest, Utterance};
    use async_trait::async_trait;

    struct MockSttBackend;
    #[async_trait]
    impl SttBackend for MockSttBackend {
        async fn transcribe(&self, _audio: &Utterance) -> Result<SttResult, VoxError> {
            Ok(SttResult {
                text: "test".to_string(),
                language: None,
                duration_ms: 0,
                processing_time_ms: 0,
            })
        }
    }

    #[allow(dead_code)]
    struct MockTtsBackend;
    #[async_trait]
    impl TtsBackend for MockTtsBackend {
        async fn synthesize(&self, _req: &TtsRequest) -> Result<TtsOutput, VoxError> {
            Ok(TtsOutput {
                audio: crate::types::AudioChunk {
                    samples: vec![],
                    sample_rate: 16000,
                    channels: 1,
                },
                duration_ms: 0,
            })
        }
    }

    #[test]
    fn test_cache_hit() {
        let cache = ModelCache::new(2);
        let key = CacheKey {
            kind: "stt".to_string(),
            identifier: "test".to_string(),
        };

        // First access - miss
        let _backend1 = cache
            .get_or_load_stt(key.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Second access - hit
        let _backend2 = cache
            .get_or_load_stt(key.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = ModelCache::new(2);

        // Add 3 models (should evict oldest)
        let key1 = CacheKey {
            kind: "stt".to_string(),
            identifier: "model1".to_string(),
        };
        let key2 = CacheKey {
            kind: "stt".to_string(),
            identifier: "model2".to_string(),
        };
        let key3 = CacheKey {
            kind: "stt".to_string(),
            identifier: "model3".to_string(),
        };

        cache
            .get_or_load_stt(key1.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();
        cache
            .get_or_load_stt(key2.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();

        // Access key1 to make it more recent than key2
        cache
            .get_or_load_stt(key1.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();

        // Add key3 - should evict key2 (LRU)
        cache
            .get_or_load_stt(key3.clone(), || Ok(Arc::new(MockSttBackend)))
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.entries, 2); // max 2 entries
    }
}
