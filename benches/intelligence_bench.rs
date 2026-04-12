//! Benchmarks for the intelligence layer.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;
use vox::intelligence::{PreferenceType, SemanticCache, SemanticCacheConfig, UserModel};

fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache");

    // Cache hit benchmark
    group.bench_function("hit", |b| {
        let cache = SemanticCache::new();
        cache.put("What is the weather?", "It's sunny.");

        b.iter(|| {
            let _ = cache.get("What is the weather?");
        });
    });

    // Cache miss benchmark
    group.bench_function("miss", |b| {
        let cache = SemanticCache::new();

        b.iter(|| {
            let _ = cache.get("What is the weather?");
        });
    });

    // Cache put benchmark
    group.bench_function("put", |b| {
        let cache = SemanticCache::new();
        let mut i = 0;

        b.iter(|| {
            cache.put(&format!("question_{}", i), "response");
            i += 1;
        });
    });

    // Normalization benchmark
    group.bench_function("normalization", |b| {
        let cache = SemanticCache::new();

        b.iter(|| {
            cache.get("What, um, is the, like, weather today?!");
        });
    });

    group.finish();
}

fn bench_cache_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_size");

    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = SemanticCacheConfig {
                max_entries: size,
                ttl: Duration::from_secs(3600),
                enable_metrics: true,
            };
            let cache = SemanticCache::with_config(config);

            // Pre-fill cache to capacity
            for i in 0..size {
                cache.put(&format!("question_{}", i), "response");
            }

            let mut i = size;
            b.iter(|| {
                cache.put(&format!("question_{}", i), "response");
                i += 1;
            });
        });
    }

    group.finish();
}

fn bench_user_model_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("user_model");

    // Set preference benchmark
    group.bench_function("set_preference", |b| {
        let model = UserModel::new();

        b.iter(|| {
            model.set_preference("user123", PreferenceType::Verbosity(5));
        });
    });

    // Get profile benchmark
    group.bench_function("get_profile", |b| {
        let model = UserModel::new();
        model.set_preference("user123", PreferenceType::Verbosity(5));

        b.iter(|| {
            let _ = model.get_profile("user123");
        });
    });

    // Record interaction benchmark
    group.bench_function("record_interaction", |b| {
        let model = UserModel::new();

        b.iter(|| {
            model.record_interaction(
                "user123",
                "What is the weather like today?",
                "It's sunny and warm.",
                true,
            );
        });
    });

    // Build system prompt modifier benchmark
    group.bench_function("build_prompt_modifier", |b| {
        let model = UserModel::new();
        model.set_preference(
            "user123",
            PreferenceType::ResponseStyle("casual".to_string()),
        );
        model.set_preference("user123", PreferenceType::Verbosity(8));

        b.iter(|| {
            let _ = model.build_system_prompt_modifier("user123");
        });
    });

    group.finish();
}

fn bench_cache_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit_rate");

    // Simulate realistic hit rates
    for hit_rate in [0.0, 0.25, 0.50, 0.75, 0.90].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", (hit_rate * 100.0) as u32)),
            hit_rate,
            |b, &hit_rate| {
                let cache = SemanticCache::new();

                // Pre-fill cache with some common queries
                let common_queries = [
                    "What's the weather?",
                    "Tell me a joke",
                    "What time is it?",
                    "How are you?",
                    "What's your name?",
                ];

                for query in common_queries.iter() {
                    cache.put(query, "response");
                }

                let mut i = 0;
                b.iter(|| {
                    let is_hit = (i as f64 / 100.0) < hit_rate;

                    if is_hit && i < common_queries.len() {
                        // Cache hit
                        let _ = cache.get(common_queries[i % common_queries.len()]);
                    } else {
                        // Cache miss
                        let _ = cache.get(&format!("unique_query_{}", i));
                    }

                    i += 1;
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_operations,
    bench_cache_sizes,
    bench_user_model_operations,
    bench_cache_hit_rate,
);
criterion_main!(benches);
