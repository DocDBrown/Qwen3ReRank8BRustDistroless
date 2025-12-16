use anyhow::{Context, Result, anyhow};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

pub trait Reranker: Send + Sync {
    // Returns Vec<Vec<f32>> to handle [Batch, Num_Classes]
    fn compute_logits(
        &self,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        shape: [usize; 2],
    ) -> Result<Vec<Vec<f32>>>;
}

struct OrtReranker {
    session: Mutex<Session>,
}

impl Reranker for OrtReranker {
    fn compute_logits(
        &self,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        shape: [usize; 2],
    ) -> Result<Vec<Vec<f32>>> {
        let input_ids_tensor = Tensor::from_array((shape, input_ids))
            .map_err(|e| anyhow!("failed to create input_ids tensor: {e}"))?
            .into_dyn();
        let attention_mask_tensor = Tensor::from_array((shape, attention_mask))
            .map_err(|e| anyhow!("failed to create attention_mask tensor: {e}"))?
            .into_dyn();

        let mut inputs = std::collections::HashMap::new();
        inputs.insert("input_ids".to_string(), input_ids_tensor);
        inputs.insert("attention_mask".to_string(), attention_mask_tensor);

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("failed to acquire session lock"))?;

        let outputs = session
            .run(inputs)
            .map_err(|e| anyhow!("inference failed: {e}"))?;

        let output_tensor = if let Some(v) = outputs.get("logits") {
            v
        } else {
            let first_key = outputs
                .keys()
                .next()
                .ok_or_else(|| anyhow!("Model produced no outputs"))?;
            outputs
                .get(first_key)
                .ok_or_else(|| anyhow!("Failed to retrieve first output"))?
        };

        let output_shape = output_tensor.shape();
        let scores_raw = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|_| anyhow!("Output is not f32"))?;
        let flat_scores = scores_raw.1;

        let batch_size = shape[0];
        let mut batch_logits = Vec::with_capacity(batch_size);

        if output_shape.len() == 2 {
            let num_classes = output_shape[1] as usize;
            for i in 0..batch_size {
                let start = i * num_classes;
                let end = start + num_classes;
                if end <= flat_scores.len() {
                    batch_logits.push(flat_scores[start..end].to_vec());
                }
            }
        } else if output_shape.len() == 1 {
            for &score in flat_scores {
                batch_logits.push(vec![score]);
            }
        } else {
            return Err(anyhow!("Unexpected output shape: {:?}", output_shape));
        }

        Ok(batch_logits)
    }
}

#[derive(Clone)]
struct AppState {
    tokenizer: Arc<Tokenizer>,
    reranker: Arc<dyn Reranker>,
    model_name: String,
    max_length: usize,
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: Option<bool>,
}

#[derive(Serialize, Clone)]
struct RerankResult {
    index: usize,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    document: Option<String>,
}

#[derive(Serialize)]
struct RerankResponse {
    object: &'static str,
    model: String,
    data: Vec<RerankResult>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_target(false)
        .compact()
        .init();

    let model_path = PathBuf::from(
        std::env::var("RERANKER_MODEL_PATH").unwrap_or_else(|_| "onnx/model.onnx".to_string()),
    );
    let tokenizer_path = PathBuf::from(
        std::env::var("RERANKER_TOKENIZER_PATH")
            .unwrap_or_else(|_| "onnx/tokenizer.json".to_string()),
    );

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))
        .with_context(|| format!("path: {tokenizer_path:?}"))?;
    let tokenizer = Arc::new(tokenizer);

    #[cfg(feature = "load-dynamic")]
    {
        let dylib_path = std::env::var("ORT_DYLIB_PATH")
            .unwrap_or_else(|_| "/opt/onnxruntime/lib/libonnxruntime.so".to_string());
        ort::init_from(dylib_path)
            .commit()
            .map_err(|e| anyhow!("failed to init ORT: {e}"))?;
    }
    #[cfg(not(feature = "load-dynamic"))]
    {
        ort::init()
            .commit()
            .map_err(|e| anyhow!("failed to init ORT: {e}"))?;
    }

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&model_path)
        .map_err(|e| anyhow!("failed to load ONNX model: {e}"))?;

    info!("Model loaded successfully");

    let reranker = Arc::new(OrtReranker {
        session: Mutex::new(session),
    });

    let state = AppState {
        tokenizer,
        reranker,
        model_name: "Qwen3-Reranker-8B".to_string(),
        max_length: 512,
    };

    let app = Router::new()
        .route("/v1/rerank", post(handle_rerank))
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8982".to_string())
        .parse()?;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

async fn handle_rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, (StatusCode, String)> {
    if req.documents.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "documents must be non-empty".to_string(),
        ));
    }

    let mut encodings = Vec::with_capacity(req.documents.len());
    for doc in &req.documents {
        let input_text = format!("Query: {}\nDocument: {}", req.query, doc);
        let encoding = state
            .tokenizer
            .encode(input_text, true)
            .map_err(internal_err)?;
        encodings.push(encoding);
    }

    let batch_size = encodings.len();
    let max_len_in_batch = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);
    let max_len = std::cmp::min(max_len_in_batch, state.max_length);

    let mut input_ids = vec![0i64; batch_size * max_len];
    let mut attention_mask = vec![0i64; batch_size * max_len];

    let pad_id = state
        .tokenizer
        .token_to_id("<|endoftext|>")
        .or_else(|| state.tokenizer.token_to_id("<|im_end|>"))
        .unwrap_or(0) as i64;

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let len = std::cmp::min(ids.len(), max_len);

        for j in 0..len {
            input_ids[i * max_len + j] = ids[j] as i64;
            attention_mask[i * max_len + j] = 1;
        }
        for j in len..max_len {
            input_ids[i * max_len + j] = pad_id;
            attention_mask[i * max_len + j] = 0;
        }
    }

    let shape = [batch_size, max_len];
    let reranker = state.reranker.clone();

    let all_logits = tokio::task::spawn_blocking(move || {
        reranker.compute_logits(input_ids, attention_mask, shape)
    })
    .await
    .map_err(|e| internal_err(format!("Task join error: {e}")))?
    .map_err(internal_err)?;

    let mut results = Vec::new();
    for (i, logits) in all_logits.iter().enumerate() {
        if i >= batch_size {
            break;
        }

        info!("Doc {}: Raw Logits: {:?}", i, logits);

        let score = if logits.len() == 1 {
            1.0 / (1.0 + (-logits[0]).exp())
        } else if logits.len() == 2 {
            // FIX: Switched to Index 0 based on your logs.
            // Index 1 was "Irrelevant" (Canberra had lowest score).
            // Index 0 is "Relevant".
            let relevant_logit = logits[0];
            1.0 / (1.0 + (-relevant_logit).exp())
        } else {
            0.0
        };

        results.push(RerankResult {
            index: i,
            score,
            document: if req.return_documents.unwrap_or(false) {
                Some(req.documents[i].clone())
            } else {
                None
            },
        });
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(n) = req.top_n {
        results.truncate(n);
    }

    Ok(Json(RerankResponse {
        object: "list",
        model: state.model_name,
        data: results,
    }))
}

fn internal_err<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    error!("internal error: {e}");
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

#[cfg(test)]
mod tests_api;
#[cfg(test)]
mod tests_logic;


#[cfg(test)]
mod tests_api;
#[cfg(test)]
mod tests_logic;
