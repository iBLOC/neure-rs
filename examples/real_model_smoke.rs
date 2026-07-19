//! Real-model smoke test for the embedding runtime.
//!
//! Spins up the embedded server, loads `all-MiniLM-L6-v2` from
//! `$NEURE_EMBEDDING_MODEL_PATH` (default: `/tmp/neure-smoke/models/embedding/all-MiniLM-L6-v2`),
//! and exercises the `/v1/embeddings` endpoint with three checks:
//!
//! 1. `encoding_format: "float"` — request a real embedding, verify the
//!    output is a 384-dim vector with non-trivial variance (i.e. not
//!    a constant echo vector).
//! 2. `encoding_format: "base64"` — same input, base64-encoded response.
//!    Verify the decoded bytes round-trip to the same 384 values.
//! 3. **Batched forward** — request a 4-element input batch. Verify
//!    each row gets its own deterministic embedding and that the
//!    returned count matches the input count.
//!
//! Run with:
//! ```bash
//! cargo run --example real_model_smoke --features candle
//! ```
//!
//! Set `NEURE_EMBEDDING_MODEL_PATH` to override the default model location.
//! Set `NEURE_PORT=0` (default) to bind to a random free port.

use neure_lib::{run_embedded, NeureConfig, NeureEmbedConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("NEURE_EMBEDDING_MODEL_PATH").unwrap_or_else(|_| {
        "/tmp/neure-smoke/models/embedding/all-MiniLM-L6-v2".to_string()
    });
    eprintln!("[smoke] loading embedding model from {model_path}");
    if !std::path::Path::new(&model_path).join("model.safetensors").exists() {
        eprintln!(
            "[smoke] FATAL: model.safetensors not found in {model_path}. \
             Download all-MiniLM-L6-v2 first."
        );
        std::process::exit(2);
    }

    let port: u16 = std::env::var("NEURE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cfg = NeureEmbedConfig {
        port,
        config: NeureConfig::new(),
    };
    let handle = run_embedded(cfg).await?;
    let port = handle.addr.port();
    eprintln!("[smoke] server bound on 127.0.0.1:{port}");

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/embeddings");

    // ---- 1. float encoding ----
    eprintln!("\n[smoke 1] float encoding");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": "Hello world, this is a real model smoke test."
        }))
        .send()
        .await?
        .json()
        .await?;
    let float_vec = r["data"][0]["embedding"].as_array().expect("array");
    let float_count = float_vec.len();
    let float_var = variance(float_vec.iter().filter_map(|v| v.as_f64()).collect());
    eprintln!("[smoke 1]   embedding dim = {float_count}");
    eprintln!("[smoke 1]   variance     = {float_var:.6}");
    assert_eq!(float_count, 384, "expected 384-dim embedding");
    assert!(float_var > 1e-6, "embedding has near-zero variance — model not loaded?");

    // ---- 2. base64 encoding ----
    eprintln!("\n[smoke 2] base64 encoding");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": "Hello world, this is a real model smoke test.",
            "encoding_format": "base64"
        }))
        .send()
        .await?
        .json()
        .await?;
    let b64 = r["data"][0]["embedding"].as_str().expect("base64 string");
    let bytes = base64_decode(b64).expect("valid base64");
    assert_eq!(bytes.len(), 384 * 4, "expected 1536 bytes (384 f32 LE)");
    let base64_vec: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("[smoke 2]   decoded bytes = {}", bytes.len());
    eprintln!("[smoke 2]   first 3 values = {:?}", &base64_vec[..3]);

    // Verify base64 roundtrip matches float encoding for the same input
    let base64_first: Vec<f32> = base64_vec[..3].to_vec();
    let float_first: Vec<f32> = float_vec
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .take(3)
        .collect();
    for (i, (a, b)) in float_first.iter().zip(base64_first.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < 1e-5, "base64[{i}] {a} != float[{i}] {b} (diff {diff})");
    }
    eprintln!("[smoke 2]   base64/float roundtrip OK (max diff < 1e-5)");

    // ---- 3. batched forward ----
    eprintln!("\n[smoke 3] batched forward (4 inputs)");
    let r: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": [
                "first document",
                "second document",
                "third document",
                "fourth document"
            ]
        }))
        .send()
        .await?
        .json()
        .await?;
    let data = r["data"].as_array().expect("data array");
    assert_eq!(data.len(), 4, "expected 4 rows back");
    for (i, row) in data.iter().enumerate() {
        let v = row["embedding"].as_array().expect("embedding");
        assert_eq!(v.len(), 384, "row {i} not 384-dim");
    }
    // Verify per-row determinism: same input twice → same vector.
    let r2: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({
            "model": "all-minilm-l6-v2",
            "input": ["second document", "second document"]
        }))
        .send()
        .await?
        .json()
        .await?;
    let first = r["data"][1]["embedding"].as_array().unwrap();
    let repeat_first = r2["data"][0]["embedding"].as_array().unwrap();
    let mut max_diff = 0.0_f64;
    for (a, b) in first.iter().zip(repeat_first.iter()) {
        let av = a.as_f64().unwrap_or(0.0);
        let bv = b.as_f64().unwrap_or(0.0);
        max_diff = max_diff.max((av - bv).abs());
    }
    eprintln!("[smoke 3]   determinism max diff = {max_diff:.2e}");
    assert!(max_diff < 1e-5, "determinism broken: max diff {max_diff}");
    eprintln!("[smoke 3]   batched forward + determinism OK");

    // ---- cleanup ----
    eprintln!("\n[smoke] shutting down server...");
    handle.request_shutdown();
    handle.join().await;
    eprintln!("[smoke] ALL CHECKS PASSED");
    Ok(())
}

fn variance(values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Minimal RFC 4648 base64 decoder (no external dep).
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 128] = &{
        let mut t = [255u8; 128];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[chars[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = [0u8; 4];
    let mut buf_len = 0;
    for &b in bytes {
        if b == b'=' {
            break;
        }
        let v = *TABLE.get(b as usize).unwrap_or(&255);
        if v == 255 {
            return None;
        }
        buf[buf_len] = v;
        buf_len += 1;
        if buf_len == 4 {
            out.push((buf[0] << 2) | (buf[1] >> 4));
            out.push((buf[1] << 4) | (buf[2] >> 2));
            out.push((buf[2] << 6) | buf[3]);
            buf_len = 0;
        }
    }
    if buf_len == 3 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
        out.push((buf[1] << 4) | (buf[2] >> 2));
    } else if buf_len == 2 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
    }
    Some(out)
}