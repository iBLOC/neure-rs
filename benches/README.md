# Benchmarking neure

This directory contains Criterion benchmarks for measuring performance of neure's data-structure operations and DTO construction/sorting. Real-engine benchmarks (LLM forward passes, TTS synthesis, rerank scoring) require a separate benchmarking harness with model weights — see `examples/real_model_smoke.rs` and `examples/real_model_rerank_smoke.rs` for end-to-end smoke tests.

## Running Benchmarks

Full benchmark (sample_size=50, measurement_time=5s per benchmark):
```bash
cargo bench --bench perf
```

Quick benchmark (1 sample, 1 iteration) for CI:
```bash
cargo bench --bench perf -- --quick
```

With specific features:
```bash
cargo bench --bench perf --features candle
```

## Benchmark Categories

| Benchmark | Purpose |
|-----------|---------|
| `base64_encode_micro` | Base64 encode 384 f32 values |
| `rerank_response_[10,100,1000]` | Response construction + sorting |
| `model_info_[10,100,1000]` | JSON serialization of ModelInfo |

## Removed in 2026-06-30 (echo stub removal)

These benchmarks were removed when the `Echo*Runtime` stubs were deleted:

- `embedding_echo_size_[1,8,32,128]` — embedding batch scaling via `EchoEmbeddingRuntime`
- `embedding_float` / `embedding_base64` — encoding format overhead via `EchoEmbeddingRuntime`
- `llm_non_streaming` / `llm_streaming` — LLM streaming overhead via `EchoLlmRuntime`
- `rerank_docs_[1,10,50,100]` — rerank document count scaling via `EchoRerankRuntime`
- `tts_len_[10,100,1000]` — TTS input length via `EchoTtsRuntime`

To re-introduce comparable benchmarks, point them at a real engine implementation behind a feature gate (e.g. `#[cfg(feature = "candle")]` + `CandleEmbeddingRuntime::embed(...)`).

## CI

Recommended CI command (fast smoke test):
```bash
cargo bench --bench perf -- --quick
```

This runs each benchmark with 1 sample and 1 iteration, completing in ~30 seconds.