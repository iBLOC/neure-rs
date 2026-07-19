---
title: Environment Variables
---

# Environment Variables

neure reads its configuration from environment variables on startup. The full matrix below documents every `NEURE_*` variable, plus the model-source override variables and the third-party tool env vars.

## Server

| Variable | Default | Description |
|---|---|---|
| `NEURE_PORT` | `8083` | HTTP bind port (embedded hosts commonly override to 8085) |
| `NEURE_HOST` | `0.0.0.0` | HTTP bind host (`0.0.0.0` = all interfaces; `127.0.0.1` = loopback only) |

## Model directory

| Variable | Default | Description |
|---|---|---|
| `NEURE_MODEL_DIRS` | `~/.neure/models/` | Colon-separated list of roots scanned for downloaded model weights (use `NEURE_MODEL_DIR` for the single legacy value) |
| `NEURE_LLM_MODEL_PATH` | (unset) | Path to the LLM model directory (must contain `config.json` + `tokenizer.json` + `*.safetensors`) |
| `NEURE_TTS_MODEL_PATH` | (unset) | Path to the TTS model directory (VoxCpm: `config.json` + `tokenizer.json` + `model.mpk` + `audiovae.mpk`) |
| `NEURE_ASR_MODEL_PATH` | (unset) | Path to the ASR model directory (Whisper) |
| `NEURE_RERANK_MODEL_PATH` | (unset) | Path to the Rerank model directory (BGE/mxbai/jina; not used for cohere) |
| `NEURE_EMBEDDING_MODEL_PATH` | (unset) | Path to the Embedding model directory (MiniLM BERT) |
| `NEURE_VISION_MODEL_PATH` | (unset) | Path to the Vision model directory (YOLOv8: `config.json` + `yolov8n.safetensors`) |

## Engine selection

| Variable | Default | Description |
|---|---|---|
| `NEURE_LLM_RUNTIME` | (unset) | `candle` \| `litert` \| `mistralrs` — which LLM engine to load |
| `NEURE_MISTRALRS_ISQ` | `q4_0` | `q4_0` \| `q8_0` \| `none` — in-situ quantization applied by mistral.rs at load time |
| `NEURE_EMBEDDING_RUNTIME` | (unset) | `candle` — which Embedding engine to load |
| `NEURE_TTS_RUNTIME` | (unset) | `voxcpm` (requires `--features voxcpm`) |
| `NEURE_ASR_RUNTIME` | (unset) | `candle` (Whisper) |
| `NEURE_RERANK_RUNTIME` | (unset) | `candle` (BGE-reranker, deprecated alias for `bge`) \| `bge` \| `mxbai` \| `jina` \| `cohere` |
| `NEURE_COHERE_API_KEY` | (unset) | Cohere API key (required when `NEURE_RERANK_RUNTIME=cohere`) |
| `NEURE_COHERE_ENDPOINT` | `https://api.cohere.com` | Cohere API base URL |
| `NEURE_COHERE_MODEL` | `rerank-english-v3.0` | Cohere model id (e.g. `rerank-multilingual-v3.0`) |
| `NEURE_VISION_RUNTIME` | (unset) | `candle-yolo` \| `candle-rtdetr` \| `candle-detr` \| `ort` \| `ultralytics` — which Vision engine + model family to load |

## Model source registry

| Variable | Default | Description |
|---|---|---|
| `NEURE_DEFAULT_SOURCE` | `modelscope` | Default source id used by `POST /v1/models/pull` and `parse_reference` when the reference has no `<source>:` prefix |
| `NEURE_LLM_SOURCE` | (unset) | Per-engine source id override (LLM) |
| `NEURE_TTS_SOURCE` | (unset) | Per-engine source id override (TTS) |
| `NEURE_ASR_SOURCE` | (unset) | Per-engine source id override (ASR) |
| `NEURE_RERANK_SOURCE` | (unset) | Per-engine source id override (Rerank) |
| `NEURE_MODEL_SOURCE_LLM_<ID>` | (unset) | Per-model source id override (LLM); use underscores for hyphens in model id |
| `NEURE_MODEL_SOURCE_TTS_<ID>` | (unset) | Per-model source id override (TTS) |
| `NEURE_MODEL_SOURCE_ASR_<ID>` | (unset) | Per-model source id override (ASR) |
| `NEURE_MODEL_SOURCE_RERANK_<ID>` | (unset) | Per-model source id override (Rerank) |
| `NEURE_SOURCE_<ID>_ENDPOINT` | (unset) | Base URL for the named source (e.g. `NEURE_SOURCE_HUGGINGFACE_ENDPOINT=https://my-mirror.com`) |
| `NEURE_HUGGINGFACE_CLI` | (auto, on PATH) | Path to `huggingface-cli` binary for `HuggingFaceSource` (also used by `hf-mirror`) |
| `NEURE_MODELSCOPE_CLI` | (auto, on PATH) | Path to `modelscope-cli` binary for `ModelScopeSource` |

## Authentication

| Variable | Description |
|---|---|
| `HF_TOKEN` | Bearer token sent on every HuggingFace / hf-mirror HTTP download. Empty / unset is fine — `reqwest` simply omits the header. Used to un-gate private repos. |
| `MODELSCOPE_TOKEN` | Bearer token sent on every ModelScope HTTP download. Same semantics as `HF_TOKEN`. |

## Networking

| Variable | Description |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | Used by `reqwest` for outbound HTTP (model downloads). `ALL_PROXY` is intentionally skipped (may be SOCKS, which `reqwest` can't handle without the `socks` feature). |
| `RUST_LOG` | Standard `tracing` filter (e.g. `RUST_LOG=neure=debug,info`) |

## Hardware

| Variable | Description |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Standard CUDA device selection (passed through to `cudarc`) |
| `ORT_CUDA` | If set, ONNX runtime uses CUDA; otherwise CPU (10x slower for large models) |
| `ORT_TENSORRT` | If set, ONNX runtime uses TensorRT (NVIDIA-specific, faster than CUDA) |

## Caching

| Variable | Default | Description |
|---|---|---|
| `NEURE_HUGGINGFACE_ENDPOINT` | `https://huggingface.co` | Custom HuggingFace endpoint (e.g. enterprise proxy) |
| `NEURE_MODELSCOPE_ENDPOINT` | `https://www.modelscope.cn` | Custom ModelScope endpoint (e.g. internal mirror) |
| `NEURE_SOURCE_HUGGINGFACE_TOKEN` | (unset) | Bearer token for the `huggingface` source endpoint |
| `NEURE_SOURCE_MODELSCOPE_TOKEN` | (unset) | Bearer token for the `modelscope` source endpoint |

## Default models

The first time neure starts, it registers a default model per capability based on the installed features. To change the default, set `NEURE_DEFAULT_LLM_MODEL` etc.

| Variable | Default | Description |
|---|---|---|
| `NEURE_DEFAULT_LLM_MODEL` | (system) | Model id to use when `model: ""` is requested |
| `NEURE_DEFAULT_TTS_MODEL` | (system) | Model id for TTS requests |
| `NEURE_DEFAULT_ASR_MODEL` | (system) | Model id for ASR requests |
| `NEURE_DEFAULT_RERANK_MODEL` | (system) | Model id for rerank requests |
| `NEURE_DEFAULT_EMBEDDING_MODEL` | (system) | Model id for embedding requests |
| `NEURE_DEFAULT_VISION_MODEL` | (system) | Model id for vision requests |

## Examples

### Minimal desktop config (Tauri app)

```bash
export NEURE_PORT=8085
export NEURE_HOST=127.0.0.1
export NEURE_MODEL_DIRS="$HOME/.neure/models"
export NEURE_LLM_RUNTIME=candle
export NEURE_DEFAULT_SOURCE=huggingface
```

### Production server config (high throughput)

```bash
export NEURE_PORT=8080
export NEURE_HOST=0.0.0.0
export NEURE_LLM_RUNTIME=mistralrs
export NEURE_MISTRALRS_ISQ=q8_0
export NEURE_RERANK_RUNTIME=cohere
export NEURE_COHERE_API_KEY=...
export NEURE_HUGGINGFACE_TOKEN=hf_...
export NEURE_DEFAULT_SOURCE=modelscope
export RUST_LOG=neure=info,tower_http=warn
```

### Edge / mobile config (low memory)

```bash
export NEURE_PORT=8085
export NEURE_HOST=127.0.0.1
export NEURE_LLM_RUNTIME=litert
export NEURE_DEFAULT_SOURCE=modelscope
export RUST_LOG=warn
```

## Validation

neure reads all `NEURE_*` env vars at startup. If a value is invalid (e.g. unknown runtime name, malformed source URL), you'll see a clear error in stderr:

```
Error: invalid value for NEURE_LLM_RUNTIME: "candl" (expected one of: candle, litert, mistralrs)
Error: invalid value for NEURE_RERANK_RUNTIME: "coher" (expected one of: candle, bge, mxbai, jina, cohere)
```

The startup fails fast — you can't accidentally boot with a bad config.

## Next steps

- [Embed neure into a Rust Host](/howto/embed-into-host) — minimal host setup
- [Capabilities](/concepts/capabilities) — what each capability does
- [OpenAI-compatible API](/reference/api) — full route reference
