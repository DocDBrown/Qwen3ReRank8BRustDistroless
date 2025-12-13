use super::*;
use axum::extract::State;
use axum::Json;
use axum::http::StatusCode;

async fn get_api_test_state() -> AppState {
    crate::tests_logic::get_test_state().await
}

#[tokio::test]
async fn test_rerank_rejects_empty_document_list() {
    let state = get_api_test_state().await;
    let req = RerankRequest {
        query: "test".into(),
        documents: vec![],
        top_n: None,
        return_documents: None,
    };

    let result = handle_rerank(State(state), Json(req)).await;

    match result {
        Ok(_) => panic!("Should have failed"),
        Err((code, msg)) => {
            assert_eq!(code, StatusCode::BAD_REQUEST);
            assert_eq!(msg, "documents must be non-empty");
        }
    }
}

#[tokio::test]
async fn test_tokenization_truncates_inputs_exceeding_max_length() {
    let state = get_api_test_state().await;
    // Create a long document. The dummy tokenizer splits by space.
    let long_doc = "word ".repeat(200); 
    
    let req = RerankRequest {
        query: "test".into(),
        documents: vec![long_doc],
        top_n: None,
        return_documents: None,
    };

    let result = handle_rerank(State(state), Json(req)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_tokenization_pads_batch_to_longest_sequence() {
    let state = get_api_test_state().await;
    let req = RerankRequest {
        query: "test".into(),
        documents: vec![
            "short".into(),
            "this is a much longer document".into()
        ],
        top_n: None,
        return_documents: None,
    };

    let result = handle_rerank(State(state), Json(req)).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().0.data.len(), 2);
}

#[tokio::test]
async fn test_tokenization_handles_unicode_and_special_characters() {
    let state = get_api_test_state().await;
    let req = RerankRequest {
        query: "Hello".into(),
        documents: vec![
            "Hello world".into(),
            "你好".into(), 
            "😊".into(),
        ],
        top_n: None,
        return_documents: None,
    };

    let result = handle_rerank(State(state), Json(req)).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().0.data.len(), 3);
}

#[tokio::test]
async fn test_concurrency_handles_simultaneous_requests_thread_safety() {
    let state = get_api_test_state().await;
    let mut handles = Vec::new();

    for i in 0..10 {
        let state_clone = state.clone();
        handles.push(tokio::spawn(async move {
            let req = RerankRequest {
                query: format!("query {}", i),
                documents: vec!["doc A".into(), "doc B".into()],
                top_n: None,
                return_documents: None,
            };
            handle_rerank(State(state_clone), Json(req)).await
        }));
    }

    for handle in handles {
        let res = handle.await.expect("Task panicked");
        assert!(res.is_ok());
    }
}