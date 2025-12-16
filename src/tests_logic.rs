use super::*;
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
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = shape[0];
        // Return deterministic logits for testing sorting.
        // We return decreasing values: 10.0, 9.0, 8.0...
        // We wrap them in Vec<Vec<f32>> to match the trait.
        let logits: Vec<Vec<f32>> = (0..batch_size).map(|i| vec![10.0 - (i as f32)]).collect();
        Ok(logits)
    }
}

// Helper to create a safe test state without loading ONNX libs or files
pub(crate) async fn get_test_state() -> AppState {
    static STATE: OnceLock<AppState> = OnceLock::new();

    STATE
        .get_or_init(|| {
            // Create a dummy tokenizer in memory (no file I/O)
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

            tokenizer.add_special_tokens(&[
                tokenizers::AddedToken::from("[UNK]", true),
                tokenizers::AddedToken::from("<|endoftext|>", true),
            ]);

            AppState {
                tokenizer: Arc::new(tokenizer),
                reranker: Arc::new(MockReranker),
                model_name: "Test-Model".to_string(),
                max_length: 128,
            }
        })
        .clone()
}
