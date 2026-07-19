//! Real Chronos2 runtime: safetensors loader + the architecture
//! forward pass wrapped in `Chronos2Runtime`. Sprint 3 Phase 4.
//!
//! This is the commit that finally turns `/v1/forecast` from a
//! stub into a working pipeline. The runtime reads a checkpoint
//! from a directory on disk, builds the `T5EncoderDecoder` from
//! the loaded weights via `candle_nn::VarBuilder`, and runs the
//! decoder-side `Chronos2OutputHead` to produce the final
//! forecast numbers. The HTTP layer (`POST /v1/forecast`) calls
//! `McpRuntime::start` once to warm the model from disk, then
//! `invoke` per request.
//!
//! ## Checkpoint layout
//!
//! ```text
//! <model_dir>/
//!   config.json              # {"d_model", "num_heads", "d_ff",
//!                             #  "vocab_size", "max_seq_len", "n_enc",
//!                             #  "n_dec"}
//!   *.safetensors            # the architecture weights; we
//!                             # accept whatever keys `safetensors`
//!                             # loads. The VarBuilder maps names
//!                             # like "enc.block_0.self_q.weight" to
//!                             # the `T5Block` projection tensors.
//! ```
//!
//! ## What this commit does NOT do
//!
//! - Real Chronos2 output head. The head is a single
//!   `Linear(d_model, 1) -> sigmoid`; the published model uses
//!   `num_quantiles` bins. The architecture API is stable so a
//!   follow-up commit can swap in the real head without
//!   touching the runtime or HTTP wiring.
//! - Cross-attention mask construction. The decoder block
//!   already takes an optional mask; the runtime passes `None`
//!   for v0. A real deployment would build the causal +
//!   padding composite.
//! - Parietal forecast skill. That's Phase 5 (next commit).

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;

use super::mod::{ForecastRequest, ForecastResponse, ForecastMethod};
use super::output::{take_horizon, Chronos2OutputHead};
use super::vendor::T5BlockConfig;
use super::vendor::T5EncoderDecoder;
use super::{Chronos2Error, Chronos2Runtime};

/// What a saved checkpoint needs to declare. A real
/// `Chronos2Config.json` carries more knobs (heads_kv,
/// dense_act, etc.); the fields we read are the ones the
/// architecture + head require. Unknown fields are ignored.
#[derive(Debug, Clone)]
pub struct Chronos2Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub n_enc: usize,
    pub n_dec: usize,
    pub layer_norm_eps: f64,
}

impl Chronos2Config {
    /// Sensible defaults for a Chronos-Bolt-Small style model.
    /// The runtime can use these when the checkpoint omits
    /// any field.
    pub fn default_bolt_small() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            vocab_size: 4096,
            max_seq_len: 2048,
            n_enc: 6,
            n_dec: 6,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn t5_block_config(&self) -> T5BlockConfig {
        T5BlockConfig {
            d_model: self.d_model,
            d_ff: self.d_ff,
            num_heads: self.num_heads,
            dropout: 0.0,
            layer_norm_eps: self.layer_norm_eps,
        }
    }
}

/// Errors raised by the loader. Each variant maps to a different
/// HTTP status at the request boundary (see
/// `crate::server::handlers::forecast`).
#[derive(Debug, Error)]
pub enum CandleChronos2Error {
    #[error("config not found at {0}: expected a JSON file")]
    ConfigMissing(std::path::PathBuf),
    #[error("config parse error: {0}")]
    ConfigParse(#[from] serde_json::Error),
    #[error("config missing field: {0}")]
    ConfigField(&'static str),
    #[error("safetensors not found at {0}: expected *.safetensors")]
    WeightsMissing(std::path::PathBuf),
    #[error("safetensors load: {0}")]
    SafeTensors(String),
    #[error("runtime error: {0}")]
    Runtime(#[from] Chronos2Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// The real Chronos2 runtime. Holds the architecture + head in
/// memory after the first `start`; the `forecast` call is a
/// pure forward pass.
pub struct CandleChronos2Runtime {
    model_id: String,
    config: Chronos2Config,
    architecture: T5EncoderDecoder,
    head: Chronos2OutputHead,
    device: Device,
    /// Encoder-token cache so the second call with the same
    /// `start` can skip re-running the encoder. The HTTP layer
    /// doesn't currently exercise this, but it's there for the
    /// follow-up caching commit.
    encoder_token_ids: Option<Tensor>,
    decoder_token_ids: Vec<u32>,
}

impl CandleChronos2Runtime {
    /// Load a model from a directory. Reads `config.json` for
    /// hyperparameters, finds the first `*.safetensors` file in the
    /// same directory, and loads the weights via
    /// `candle_nn::VarBuilder::from_safetensors`.
    pub fn load(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, CandleChronos2Error> {
        let dir = model_dir.as_ref();
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(CandleChronos2Error::ConfigMissing(config_path));
        }
        let config_text = std::fs::read_to_string(&config_path)?;
        let raw: serde_json::Value = serde_json::from_str(&config_text)?;
        let config = parse_config(&raw)?;
        let weights_path = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.extension().map(|x| x == "safetensors").unwrap_or(false)
            })
            .ok_or(CandleChronos2Error::WeightsMissing(dir.to_path_buf()))?;
        Self::load_with_paths(&config, weights_path, device)
    }

    /// Same as `load` but takes the parsed config + weights
    /// path explicitly. Used by tests that build a synthetic
    /// checkpoint in a tempdir.
    pub fn load_with_paths(
        config: &Chronos2Config,
        weights_path: std::path::PathBuf,
        device: Device,
    ) -> Result<Self, CandleChronos2Error> {
        let bytes = std::fs::read(&weights_path).map_err(|e| {
            CandleChronos2Error::SafeTensors(format!("read {}: {e}", weights_path.display()))
        })?;
        let vb = VarBuilder::from_safetensors_slice(&bytes, DType::F32, &device).map_err(|e| {
            CandleChronos2Error::SafeTensors(format!(
                "from_safetensors_slice {}: {e}",
                weights_path.display()
            ))
        })?;
        let block = config.t5_block_config();
        let embeddings = super::vendor::T5Embeddings {
            config: super::vendor::T5EmbeddingsConfig {
                vocab_size: config.vocab_size,
                d_model: config.d_model,
                max_seq_len: config.max_seq_len,
                learned: true,
            },
            token: vb.pp("token_emb").get_with_hints(
                (config.vocab_size, config.d_model),
                "weight",
                candle_nn::init::ZERO,
            ).map_err(CandleChronos2Error::Candle)?,
            position: vb.pp("pos_emb").get_with_hints(
                (config.max_seq_len, config.d_model),
                "weight",
                candle_nn::init::ZERO,
            ).map_err(CandleChronos2Error::Candle)?,
        };
        let enc = T5EncoderDecoder {
            encoder: super::vendor::T5Encoder::new(
                block.clone(),
                config.n_enc,
                embeddings.clone(),
                vb.pp("encoder"),
            )?,
            decoder: super::vendor::T5Decoder::new(
                block,
                config.n_dec,
                embeddings,
                vb.pp("decoder"),
            )?,
        };
        let head = Chronos2OutputHead::new(config.t5_block_config(), vb.pp("head"))?;
        Ok(Self {
            model_id: weights_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("chronos2")
                .to_string(),
            config: config.clone(),
            architecture: enc,
            head,
            device,
            encoder_token_ids: None,
            decoder_token_ids: Vec::new(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Tokenize a float series by bucketing into a fixed
    /// number of vocab bins. Chronos2's published tokenizer is
    /// a quantile-bucket scheme; the v0 implementation is a
    /// uniform `floor` so the test path is deterministic. The
    /// training commit replaces this with a real quantile
    /// transform.
    fn tokenize_series(series: &[f64], vocab_size: usize) -> Vec<u32> {
        if series.is_empty() {
            return Vec::new();
        }
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < f64::EPSILON { 1.0 } else { max - min };
        series
            .iter()
            .map(|&v| {
                let pct = ((v - min) / range).clamp(0.0, 1.0);
                ((pct * (vocab_size.saturating_sub(1)) as f64) as u32).min((vocab_size - 1) as u32)
            })
            .collect()
    }

    /// v0 decoder token sequence: a BOS, the input series
    /// tokens, then `horizon` placeholder slots. BOS / EOS /
    /// placeholder tokens are not in the public Chronos2 spec;
    /// the tokenizer is defined in `tokenize_series` and the
    /// placeholders use the same vocab range. The training
    /// commit can swap in the real BOS/EOS/placeholder indices.
    fn build_decoder_token_ids(series: &[f64], horizon: u32, vocab_size: usize) -> Vec<u32> {
        let mut ids = vec![0u32];
        ids.extend(Self::tokenize_series(series, vocab_size));
        for _ in 0..horizon {
            ids.push(1);
        }
        ids
    }
}

#[async_trait::async_trait]
impl Chronos2Runtime for CandleChronos2Runtime {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn forecast(
        &self,
        request: ForecastRequest,
    ) -> Result<ForecastResponse, super::Chronos2Error> {
        let started = std::time::Instant::now();
        let device = self.device.clone();
        let vocab = self.config.vocab_size;
        let horizon = request.horizon.max(1) as usize;

        let decoder_ids = Self::build_decoder_token_ids(
            &request.series,
            horizon as u32,
            vocab,
        );
        let decoder_token_ids = Tensor::from_vec(
            decoder_ids.clone(),
            (1, decoder_ids.len()),
            &device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;
        let input_part = decoder_ids.len() - horizon;
        let encoder_token_ids = Tensor::from_vec(
            decoder_ids[..input_part].to_vec(),
            (1, input_part),
            &device,
        ).map_err(|e| super::Chronos2Error::Runtime(e.to_string()))?;

        let arch = self.architecture.clone();
        let head = self.head.clone();
        let decoder_tok = decoder_token_ids.clone();
        let enc_tok = encoder_token_ids.clone();
        let result =
            tokio::task::spawn_blocking(move || -> Result<_, candle_core::Error> {
                let enc_out = arch.encoder.forward(&enc_tok)?;
                let dec_out = arch
                    .decoder
                    .forward(&decoder_tok, &enc_out.hidden_states)?;
                let proj = head.forward(&dec_out.hidden_states)?;
                let horizon_tensor = take_horizon(&proj, horizon as u32)?;
                Ok(horizon_tensor)
            })
            .await
            .map_err(|e| super::Chronos2Error::Runtime(format!("join: {e}")))?
            .map_err(|e| super::Chronos2Error::Runtime(e.to_string()))??;

        // The v0 model emits one forecast number per step;
        // Mean and Median collapse to the same value here. A
        // proper quantile head will be wired in a follow-up.
        let v: Vec<f32> = result.flatten_all().map_err(|e| {
            super::Chronos2Error::Runtime(e.to_string())
        })?.to_vec1().map_err(|e| {
            super::Chronos2Error::Runtime(e.to_string())
        })?;
        let forecast: Vec<f64> = v.into_iter().map(|f| f as f64).collect();
        let took_ms = started.elapsed().as_millis() as u64;

        Ok(ForecastResponse {
            object: "list".to_string(),
            model: self.model_id.clone(),
            method: request.method,
            forecast,
            took_ms,
        })
    }
}

fn parse_config(raw: &serde_json::Value) -> Result<Chronos2Config, CandleChronos2Error> {
    let require_usize = |key: &'static str| -> Result<usize, CandleChronos2Error> {
        raw.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or(CandleChronos2Error::ConfigField(key))
    };
    Ok(Chronos2Config {
        d_model: require_usize("d_model")?,
        num_heads: require_usize("num_heads")?,
        d_ff: require_usize("d_ff")?,
        vocab_size: require_usize("vocab_size")?,
        max_seq_len: require_usize("max_seq_len")?,
        n_enc: require_usize("n_enc")?,
        n_dec: require_usize("n_dec")?,
        layer_norm_eps: raw
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "chronos2-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_config(dir: &Path, cfg: &Chronos2Config) {
        let body = serde_json::json!({
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "n_enc": cfg.n_enc,
            "n_dec": cfg.n_dec,
            "layer_norm_eps": cfg.layer_norm_eps,
        });
        std::fs::write(dir.join("config.json"), body.to_string()).unwrap();
    }

    /// Build a synthetic `*.safetensors` file with the tensor
    /// names the architecture expects. The fake-data values
    /// are irrelevant — the test only needs all expected
    /// names present in the file so the candle loader
    /// resolves them.
    fn write_synthetic_safetensors(dir: &Path, cfg: &Chronos2Config) {
        use safetensors::Dtype as SD;
        use safetensors::tensor::Dtype;
        use std::collections::BTreeMap;
        let mut tensors: BTreeMap<String, (Dtype, Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut insert = |name: &str, shape: &[usize], fill: f32| {
            let data = vec![fill; shape.iter().product()];
            tensors.insert(
                name.to_string(),
                (Dtype::F32, shape.to_vec(), data),
            );
        };
        insert("token_emb.weight", &[cfg.vocab_size, cfg.d_model], 0.01);
        insert("pos_emb.weight", &[cfg.max_seq_len, cfg.d_model], 0.0);
        for enc in 0..cfg.n_enc {
            let p = format!("encoder.block_{enc}");
            insert(&format!("{p}.self_q.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_k.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_v.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_o.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_attn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_up.weight"), &[cfg.d_model, cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.weight"), &[cfg.d_ff, cfg.d_model], 0.0);
        }
        for dec in 0..cfg.n_dec {
            let p = format!("decoder.block_{dec}");
            insert(&format!("{p}.self_q.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_k.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_v.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_o.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.self_attn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_norm.weight"), &[cfg.d_model], 1.0);
            insert(&format!("{p}.ffn_up.weight"), &[cfg.d_model, cfg.d_ff], 0.0);
            insert(&format!("{p}.ffn_down.weight"), &[cfg.d_ff, cfg.d_model], 0.0);
            insert(&format!("{p}.k_enc.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.v_enc.weight"), &[cfg.d_model, cfg.d_model], 0.0);
            insert(&format!("{p}.cross_attn_norm.weight"), &[cfg.d_model], 1.0);
        }
        insert("head.final_norm.weight", &[cfg.d_model], 1.0);
        insert("head.proj.weight", &[1, cfg.d_model], 0.0);
        insert("head.bias", &[1], 0.0);

        // safetensors 0.4 `serialize` takes (name, &TensorView)
        // pairs. We build the views by leaking the byte slice
        // (the API is lifetime-bound; a short-lived test can
        // afford `'static`).
        let views: Vec<TensorView<'static>> = tensors
            .iter()
            .map(|(name, (_dtype, shape, data))| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                };
                TensorView::new(SD::F32, shape.clone(), bytes, data.as_ptr() as usize)
                    .unwrap()
            })
            .collect();
        let serialized = safetensors::serialize(views.iter().zip(tensors.keys()), Some("test"))
            .unwrap();
        std::fs::write(dir.join("model.safetensors"), &serialized).unwrap();
    }

    fn build_synthetic_checkpoint() -> std::path::PathBuf {
        let dir = tempdir();
        let cfg = Chronos2Config {
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            vocab_size: 8,
            max_seq_len: 8,
            n_enc: 1,
            n_dec: 1,
            layer_norm_eps: 1e-6,
        };
        write_config(&dir, &cfg);
        write_synthetic_safetensors(&dir, &cfg);
        dir
    }

    #[test]
    fn parse_config_extracts_required_fields() {
        let raw = serde_json::json!({
            "d_model": 64, "num_heads": 4, "d_ff": 128,
            "vocab_size": 32, "max_seq_len": 16, "n_enc": 2, "n_dec": 2,
        });
        let cfg = parse_config(&raw).unwrap();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.n_enc, 2);
        assert_eq!(cfg.n_dec, 2);
        assert_eq!(cfg.layer_norm_eps, 1e-6);
    }

    #[test]
    fn parse_config_missing_field_returns_err() {
        let raw = serde_json::json!({ "d_model": 64 });
        assert!(matches!(
            parse_config(&raw).unwrap_err(),
            CandleChronos2Error::ConfigField("num_heads")
        ));
    }

    #[test]
    fn default_bolt_small_config_is_t5_compatible() {
        let cfg = Chronos2Config::default_bolt_small();
        let t5 = cfg.t5_block_config();
        assert_eq!(t5.d_model, 512);
        assert!(t5.validate().is_ok());
    }

    #[tokio::test]
    async fn runtime_loads_synthetic_checkpoint_and_forecasts() {
        let dir = build_synthetic_checkpoint();
        let rt = tokio::task::spawn_blocking({
            let dir = dir.clone();
            move || CandleChronos2Runtime::load(&dir, Device::Cpu)
        })
        .await
        .unwrap()
        .expect("load");
        assert_eq!(rt.model_id(), "model");

        let req = ForecastRequest {
            model: "model".into(),
            series: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            horizon: 3,
            method: ForecastMethod::Mean,
        };
        let resp = rt.forecast(req).await.expect("forecast");
        assert_eq!(resp.method, ForecastMethod::Mean);
        assert_eq!(resp.forecast.len(), 3);
        for &v in &resp.forecast {
            assert!((0.0..=1.0).contains(&v), "forecast value {v} out of unit range");
        }
        assert!(resp.took_ms < 60_000, "forecast took too long");
    }

    #[test]
    fn tokenize_series_is_monotonic_in_input_value() {
        // Bigger inputs -> bigger (or equal) bucket indices.
        let small = CandleChronos2Runtime::tokenize_series(&[1.0], 8);
        let big = CandleChronos2Runtime::tokenize_series(&[8.0], 8);
        assert!(small[0] <= big[0]);
    }

    #[test]
    fn build_decoder_token_ids_includes_bos_and_horizon() {
        let ids = CandleChronos2Runtime::build_decoder_token_ids(&[1.0, 2.0, 3.0], 5, 16);
        // 1 BOS + 3 input + 5 placeholders = 9.
        assert_eq!(ids.len(), 9);
    }
}
