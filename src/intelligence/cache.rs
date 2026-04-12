//! Semantic caching for LLM responses.
//!
//! Caches LLM responses based on semantic similarity of questions rather than
//! exact string matching. Uses simple hash-based lookup with TTL and LRU eviction.
//!
//! # Privacy
//!
//! All data is stored in-memory only by default. No external services are used.
//! When persistence is enabled, data is stored locally in SQLite.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A cached LLM response entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The original question that was asked.
    question: String,
    /// The cached response.
    response: String,
    /// When this entry was created.
    created_at: Instant,
    /// When this entry was last accessed.
    last_accessed: Instant,
    /// Number of times this entry has been accessed.
    hit_count: u64,
}

/// Semantic cache configuration.
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Maximum number of entries to cache.
    pub max_entries: usize,
    /// Time-to-live for cache entries.
    pub ttl: Duration,
    /// Enable detailed metrics tracking.
    pub enable_metrics: bool,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            enable_metrics: true,
        }
    }
}

/// Metrics for the semantic cache.
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of entries evicted due to TTL expiration.
    pub ttl_evictions: u64,
    /// Total number of entries evicted due to LRU.
    pub lru_evictions: u64,
    /// Average latency for cache hits (microseconds).
    pub avg_hit_latency_us: f64,
    /// Average latency for cache misses (microseconds).
    pub avg_miss_latency_us: f64,
}

impl CacheMetrics {
    /// Calculate the cache hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate the average latency reduction (hit vs miss) in microseconds.
    pub fn latency_reduction_us(&self) -> f64 {
        if self.avg_miss_latency_us > self.avg_hit_latency_us {
            self.avg_miss_latency_us - self.avg_hit_latency_us
        } else {
            0.0
        }
    }
}

/// Semantic cache for LLM responses.
///
/// Caches responses based on semantic hash of questions. Thread-safe and can be
/// shared across multiple tasks via `Arc`.
pub struct SemanticCache {
    config: SemanticCacheConfig,
    cache: Arc<RwLock<HashMap<u64, CacheEntry>>>,
    metrics: Arc<RwLock<CacheMetrics>>,
}

impl SemanticCache {
    /// Create a new semantic cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(SemanticCacheConfig::default())
    }

    /// Create a new semantic cache with custom configuration.
    pub fn with_config(config: SemanticCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Compute a semantic hash for a question.
    ///
    /// This is a simple implementation that normalizes the question and computes a hash.
    /// More sophisticated implementations could use embeddings or fuzzy matching.
    fn semantic_hash(&self, question: &str) -> u64 {
        let normalized = self.normalize_question(question);
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Normalize a question for semantic comparison.
    ///
    /// Removes punctuation, converts to lowercase, trims whitespace, and
    /// removes common filler words.
    fn normalize_question(&self, question: &str) -> String {
        let mut normalized = question.to_lowercase();

        // Remove punctuation
        normalized = normalized
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        // Remove common filler words
        let filler_words = ["um", "uh", "like", "you know", "i mean"];
        for word in filler_words {
            normalized = normalized.replace(word, " ");
        }

        // Normalize whitespace
        normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

        normalized.trim().to_string()
    }

    /// Get a cached response for a question, if available.
    ///
    /// Returns `Some(response)` if a cached entry exists and is still valid (not expired),
    /// otherwise returns `None`.
    pub fn get(&self, question: &str) -> Option<String> {
        let start = Instant::now();
        let hash = self.semantic_hash(question);

        let mut cache = self.cache.write().unwrap();

        // Check if entry exists and is not expired
        if let Some(entry) = cache.get_mut(&hash) {
            let age = entry.created_at.elapsed();

            if age < self.config.ttl {
                // Update access time and hit count
                entry.last_accessed = Instant::now();
                entry.hit_count += 1;

                let response = entry.response.clone();
                drop(cache); // Release lock before updating metrics

                // Update metrics
                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().unwrap();
                    metrics.hits += 1;
                    let latency_us = start.elapsed().as_micros() as f64;
                    metrics.avg_hit_latency_us =
                        (metrics.avg_hit_latency_us * (metrics.hits - 1) as f64 + latency_us)
                            / metrics.hits as f64;
                }

                return Some(response);
            } else {
                // Entry expired, remove it
                cache.remove(&hash);

                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().unwrap();
                    metrics.ttl_evictions += 1;
                }
            }
        }

        drop(cache); // Release lock before updating metrics

        // Cache miss
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.misses += 1;
            let latency_us = start.elapsed().as_micros() as f64;
            metrics.avg_miss_latency_us =
                (metrics.avg_miss_latency_us * (metrics.misses - 1) as f64 + latency_us)
                    / metrics.misses as f64;
        }

        None
    }

    /// Store a response in the cache.
    ///
    /// If the cache is full, evicts the least recently used entry.
    pub fn put(&self, question: &str, response: &str) {
        let hash = self.semantic_hash(question);
        let mut cache = self.cache.write().unwrap();

        // Check if we need to evict an entry
        if cache.len() >= self.config.max_entries && !cache.contains_key(&hash) {
            // Find and remove the LRU entry
            if let Some((lru_key, _)) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(k, v)| (*k, v.last_accessed))
            {
                cache.remove(&lru_key);

                if self.config.enable_metrics {
                    let mut metrics = self.metrics.write().unwrap();
                    metrics.lru_evictions += 1;
                }
            }
        }

        // Insert or update the entry
        cache.insert(
            hash,
            CacheEntry {
                question: question.to_string(),
                response: response.to_string(),
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get current cache metrics.
    pub fn metrics(&self) -> CacheMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Reset all metrics counters.
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        *metrics = CacheMetrics::default();
    }

    /// Get the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Manually evict expired entries.
    ///
    /// This is automatically done during `get` operations, but can be called
    /// manually to clean up expired entries and free memory.
    pub fn evict_expired(&self) {
        let mut cache = self.cache.write().unwrap();
        let ttl = self.config.ttl;

        let expired_keys: Vec<u64> = cache
            .iter()
            .filter(|(_, entry)| entry.created_at.elapsed() >= ttl)
            .map(|(k, _)| *k)
            .collect();

        for key in expired_keys {
            cache.remove(&key);

            if self.config.enable_metrics {
                let mut metrics = self.metrics.write().unwrap();
                metrics.ttl_evictions += 1;
            }
        }
    }
}

impl Default for SemanticCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let cache = SemanticCache::new();

        cache.put("What is the weather today?", "It's sunny.");

        let result = cache.get("What is the weather today?");
        assert_eq!(result, Some("It's sunny.".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache = SemanticCache::new();

        let result = cache.get("What is the weather today?");
        assert_eq!(result, None);
    }

    #[test]
    fn test_semantic_similarity() {
        let cache = SemanticCache::new();

        cache.put("What is the weather today?", "It's sunny.");

        // Should hit cache due to normalization
        let result = cache.get("WHAT IS THE WEATHER TODAY?");
        assert_eq!(result, Some("It's sunny.".to_string()));

        let result = cache.get("what is the weather today");
        assert_eq!(result, Some("It's sunny.".to_string()));
    }

    #[test]
    fn test_ttl_expiration() {
        let config = SemanticCacheConfig {
            max_entries: 100,
            ttl: Duration::from_millis(100),
            enable_metrics: true,
        };
        let cache = SemanticCache::with_config(config);

        cache.put("test", "response");

        // Should hit immediately
        assert!(cache.get("test").is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should miss after TTL
        assert!(cache.get("test").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let config = SemanticCacheConfig {
            max_entries: 2,
            ttl: Duration::from_secs(3600),
            enable_metrics: true,
        };
        let cache = SemanticCache::with_config(config);

        cache.put("q1", "a1");
        cache.put("q2", "a2");

        // Access q1 to make it more recently used
        let _ = cache.get("q1");

        // Add q3, should evict q2 (least recently used)
        cache.put("q3", "a3");

        assert!(cache.get("q1").is_some());
        assert!(cache.get("q2").is_none());
        assert!(cache.get("q3").is_some());
    }

    #[test]
    fn test_metrics() {
        let cache = SemanticCache::new();

        cache.put("q1", "a1");

        let _ = cache.get("q1"); // hit
        let _ = cache.get("q2"); // miss
        let _ = cache.get("q1"); // hit

        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 2);
        assert_eq!(metrics.misses, 1);
        assert!(
            (metrics.hit_rate() - 66.67).abs() < 0.01,
            "Hit rate should be ~66.67%"
        );
    }

    #[test]
    fn test_clear() {
        let cache = SemanticCache::new();

        cache.put("q1", "a1");
        cache.put("q2", "a2");

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_normalization() {
        let cache = SemanticCache::new();

        let normalized = cache.normalize_question("What, um, is the, like, weather today?!");
        assert_eq!(normalized, "what is the weather today");
    }
}
