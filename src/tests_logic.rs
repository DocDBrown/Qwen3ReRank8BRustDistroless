use super::*;
use axum::Json;
use axum::extract::State;
use std::sync::OnceLock;
use tokenizers::models::wordlevel::WordLevel;

// Mock implementation of the Reranker trait
struct MockReranker;

impl Reranker for MockReranker {
    fn compute_logits(
        &self,
        _input_ids: Vec<i64>,
        _attention_mask: Vec<i64>,
        shape: [usize; 2],
    ) -> Result<Vec<f32>> {
        let batch_size = shape[0];
        // Return deterministic logits for testing sorting.
        // We return decreasing values: 10.0, 9.0, 8.0...
        // This ensures that index 0 has the highest score, index 1 the second, etc.
        let logits: Vec<f32> = (0..batch_size).map(|i| 10.0 - (i as f32)).collect();
        Ok(logits)
    }
}

// Helper to create a safe test state without loading ONNX libs or files
pub(crate) async fn get_test_state() -> AppState {
    static STATE: OnceLock<AppState> = OnceLock::new();

    STATE
        .get_or_init(|| {
            // Create a dummy tokenizer in memory (no file I/O)
            // We need a vocabulary that includes the UNK token and the PAD token.
            let mut vocab = std::collections::HashMap::new();
            vocab.insert("[UNK]".to_string(), 0);
            vocab.insert("<|endoftext|>".to_string(), 1);

            // Initialize WordLevel model with the vocab and UNK token
            let model = WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".to_string())
                .build()
                .expect("Failed to build WordLevel model");

            let mut tokenizer = Tokenizer::new(model);

            // Register special tokens so they are handled correctly
            // This ensures <|endoftext|> is treated as a single token and not split
            tokenizer.add_special_tokens(&[
                tokenizers::AddedToken::from("[UNK]", true),
                tokenizers::AddedToken::from("<|endoftext|>", true),
            ]);

            AppState {
                tokenizer: Arc::new(tokenizer),
                reranker: Arc::new(MockReranker), // Use Mock
                model_name: "Test-Model".to_string(),
                max_length: 128,
            }
        })
        .clone()
}

#[tokio::test]
async fn test_rerank_happy_path_sorting_correctness() {
    let state = get_test_state().await;

    let req = RerankRequest {
        query: "test".to_string(),
        documents: vec![
            "Doc A".to_string(), // Mock returns logit 10.0 -> Score ~0.999
            "Doc B".to_string(), // Mock returns logit 9.0  -> Score ~0.999 (but lower)
            "Doc C".to_string(), // Mock returns logit 8.0
        ],
        top_n: None,
        return_documents: Some(true),
    };

    let response = handle_rerank(State(state), Json(req)).await.unwrap().0;

    // Check sorting
    assert_eq!(response.data[0].index, 0);
    assert_eq!(response.data[1].index, 1);
    assert_eq!(response.data[2].index, 2);

    // Verify scores are descending
    for i in 0..response.data.len() - 1 {
        assert!(response.data[i].score >= response.data[i + 1].score);
    }
}

#[tokio::test]
async fn test_rerank_respects_top_n_parameter() {
    let state = get_test_state().await;
    let req = RerankRequest {
        query: "test".to_string(),
        documents: vec!["A".into(), "B".into(), "C".into(), "D".into()],
        top_n: Some(2),
        return_documents: None,
    };

    let response = handle_rerank(State(state), Json(req)).await.unwrap().0;
    assert_eq!(response.data.len(), 2);
}

#[tokio::test]
async fn test_rerank_handles_top_n_larger_than_batch_size() {
    let state = get_test_state().await;
    let req = RerankRequest {
        query: "test".to_string(),
        documents: vec!["A".into(), "B".into()],
        top_n: Some(10),
        return_documents: None,
    };

    let response = handle_rerank(State(state), Json(req)).await.unwrap().0;
    assert_eq!(response.data.len(), 2);
}

#[tokio::test]
async fn test_rerank_respects_return_documents_boolean() {
    let state = get_test_state().await;

    let req_false = RerankRequest {
        query: "q".into(),
        documents: vec!["d1".into()],
        top_n: None,
        return_documents: Some(false),
    };
    let res_false = handle_rerank(State(state.clone()), Json(req_false))
        .await
        .unwrap()
        .0;
    assert!(res_false.data[0].document.is_none());

    let req_true = RerankRequest {
        query: "q".into(),
        documents: vec!["d1".into()],
        top_n: None,
        return_documents: Some(true),
    };
    let res_true = handle_rerank(State(state), Json(req_true)).await.unwrap().0;
    assert_eq!(res_true.data[0].document.as_deref(), Some("d1"));
}

#[tokio::test]
async fn test_inference_calculates_sigmoid_correctly() {
    let state = get_test_state().await;
    let req = RerankRequest {
        query: "test".into(),
        documents: vec!["doc".into()],
        top_n: None,
        return_documents: None,
    };

    let response = handle_rerank(State(state), Json(req)).await.unwrap().0;
    let score = response.data[0].score;

    // Mock returns 10.0. Sigmoid(10.0) is very close to 1.0
    assert!(score > 0.99);
    assert!(score <= 1.0);
}
