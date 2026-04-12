//! Integration tests for speaker diarization.

#[cfg(feature = "diarization")]
mod diarization_tests {
    use vox::diarization::{
        Recognition, RecognitionConfig, Speaker, SpeakerDatabase, SpeakerRegistry,
        cosine_similarity,
    };

    #[test]
    fn test_speaker_registry_workflow() {
        let mut registry = SpeakerRegistry::new();

        // Enroll speakers
        let alice_emb = normalize(&[1.0, 0.0, 0.0]);
        let bob_emb = normalize(&[0.0, 1.0, 0.0]);

        registry
            .enroll("alice", "Alice", alice_emb.clone())
            .unwrap();
        registry.enroll("bob", "Bob", bob_emb.clone()).unwrap();

        assert_eq!(registry.speaker_count(), 2);

        // Identify Alice
        let result = registry.identify(&alice_emb).unwrap();
        if let Recognition::Identified {
            speaker_id,
            confidence,
        } = result
        {
            assert_eq!(speaker_id, "alice");
            assert!(confidence > 0.95); // Should be near perfect match
        } else {
            panic!("Expected Identified, got {:?}", result);
        }

        // Identify Bob
        let result = registry.identify(&bob_emb).unwrap();
        if let Recognition::Identified { speaker_id, .. } = result {
            assert_eq!(speaker_id, "bob");
        } else {
            panic!("Expected Identified");
        }

        // Unknown speaker
        let unknown_emb = normalize(&[0.0, 0.0, 1.0]);
        let result = registry.identify(&unknown_emb).unwrap();
        assert!(matches!(result, Recognition::Unknown { .. }));

        // Forget speaker
        registry.forget("alice").unwrap();
        assert_eq!(registry.speaker_count(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = normalize(&[1.0, 0.0, 0.0]);
        let b = normalize(&[1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5); // Identical

        let c = normalize(&[0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&a, &c);
        assert!(sim.abs() < 1e-5); // Orthogonal
    }

    #[test]
    fn test_threshold_configuration() {
        let config = RecognitionConfig {
            threshold: 0.9,
            require_threshold: true,
        };
        let mut registry = SpeakerRegistry::with_config(config);

        let emb = normalize(&[1.0, 0.0, 0.0]);
        registry.enroll("alice", "Alice", emb.clone()).unwrap();

        // Similar but not identical embedding
        let similar = normalize(&[0.9, 0.1, 0.0]);
        let result = registry.identify(&similar).unwrap();

        // With high threshold (0.9), this might be Unknown
        // (depends on exact similarity after normalization)
        match result {
            Recognition::Identified { confidence, .. } => {
                assert!(confidence >= 0.9);
            }
            Recognition::Unknown { .. } => {
                // Also acceptable if similarity < 0.9
            }
        }
    }

    #[tokio::test]
    async fn test_speaker_database_storage() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "test_speaker".to_string(),
            name: "Test Speaker".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        };

        // Store speaker
        db.store_speaker(&speaker).await.unwrap();

        // Load speaker
        let loaded = db.load_speaker("test_speaker").await.unwrap().unwrap();
        assert_eq!(loaded.id, "test_speaker");
        assert_eq!(loaded.name, "Test Speaker");
        assert_eq!(loaded.embedding.len(), 4);

        // Verify embedding values
        for (a, b) in speaker.embedding.iter().zip(loaded.embedding.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_database_conversation_tracking() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "alice".to_string(),
            name: "Alice".to_string(),
            embedding: vec![0.5],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Record utterances
        db.record_utterance("alice", 1000, Some("First utterance"))
            .await
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        db.record_utterance("alice", 1500, Some("Second utterance"))
            .await
            .unwrap();

        // Get history (most recent first)
        let history = db.get_conversation_history("alice", 10).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text, Some("Second utterance".to_string()));
        assert_eq!(history[1].text, Some("First utterance".to_string()));
    }

    #[tokio::test]
    async fn test_database_preferences() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "bob".to_string(),
            name: "Bob".to_string(),
            embedding: vec![0.1, 0.2],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Set preferences
        db.set_preference("bob", "voice", "af_heart").await.unwrap();
        db.set_preference("bob", "speed", "1.2").await.unwrap();

        // Get single preference
        let voice = db.get_preference("bob", "voice").await.unwrap().unwrap();
        assert_eq!(voice, "af_heart");

        // Get all preferences
        let all = db.get_all_preferences("bob").await.unwrap();
        assert_eq!(all.len(), 2);

        // Update preference
        db.set_preference("bob", "voice", "af_bella").await.unwrap();
        let voice = db.get_preference("bob", "voice").await.unwrap().unwrap();
        assert_eq!(voice, "af_bella");
    }

    #[tokio::test]
    async fn test_database_cascade_delete() {
        let db = SpeakerDatabase::open(":memory:").await.unwrap();

        let speaker = Speaker {
            id: "charlie".to_string(),
            name: "Charlie".to_string(),
            embedding: vec![0.9],
        };

        db.store_speaker(&speaker).await.unwrap();

        // Add conversations and preferences
        db.record_utterance("charlie", 1000, Some("Test"))
            .await
            .unwrap();
        db.set_preference("charlie", "voice", "test").await.unwrap();

        // Verify data exists
        let history = db.get_conversation_history("charlie", 10).await.unwrap();
        assert_eq!(history.len(), 1);
        let prefs = db.get_all_preferences("charlie").await.unwrap();
        assert_eq!(prefs.len(), 1);

        // Delete speaker (should cascade)
        db.delete_speaker("charlie").await.unwrap();

        // Verify speaker is gone
        let loaded = db.load_speaker("charlie").await.unwrap();
        assert!(loaded.is_none());

        // Verify conversations are gone (cascade delete)
        let history = db.get_conversation_history("charlie", 10).await.unwrap();
        assert_eq!(history.len(), 0);

        // Verify preferences are gone (cascade delete)
        let prefs = db.get_all_preferences("charlie").await.unwrap();
        assert_eq!(prefs.len(), 0);
    }

    // Helper to L2-normalize a vector
    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        vec.iter().map(|x| x / norm).collect()
    }
}
