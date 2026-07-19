use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::ContentBlock;
use super::sampling::{CacheControl, SamplingParams};
use super::tool::{ToolCall, ToolDefinition};
use super::usage::{StopReason, UsageInfo};

pub enum CanonicalRequest {
    Llm(CanonicalLlmRequest),
    Tts(CanonicalTtsRequest),
    Asr(CanonicalAsrRequest),
    Rerank(CanonicalRerankRequest),
    Embedding(CanonicalEmbeddingRequest),
    Vision(CanonicalVisionRequest),
}

pub enum CanonicalResponse {
    Llm(CanonicalLlmResponse),
    Tts(CanonicalTtsResponse),
    Asr(CanonicalAsrResponse),
    Rerank(CanonicalRerankResponse),
    Embedding(CanonicalEmbeddingResponse),
    Vision(CanonicalVisionResponse),
}

pub enum CanonicalStreamEvent {
    Llm(CanonicalLlmStreamEvent),
}

#[derive(Debug, Clone, Default)]
pub struct CanonicalTtsRequest { pub model: String, pub text: String }
#[derive(Debug, Clone, Default)]
pub struct CanonicalAsrRequest { pub model: String, pub audio_bytes: Vec<u8> }
#[derive(Debug, Clone, Default)]
pub struct CanonicalRerankRequest { pub model: String, pub query: String, pub documents: Vec<String> }
#[derive(Debug, Clone, Default)]
pub struct CanonicalEmbeddingRequest { pub model: String, pub inputs: Vec<String> }
#[derive(Debug, Clone, Default)]
pub struct CanonicalVisionRequest {
    pub model: String,
    pub task: String,                   // "detect" | "classify" | "segment" | "pose"
    pub image_bytes: Vec<u8>,           // raw image bytes (PNG/JPEG/WebP)
    pub image_format: String,           // "png" | "jpeg" | "webp"
    pub confidence_threshold: Option<f32>,
    pub iou_threshold: Option<f32>,
    pub max_detections: Option<u32>,
    pub classes: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Default)]
pub struct CanonicalTtsResponse { pub pcm_bytes: Vec<u8>, pub sample_rate: u32, pub channels: u32 }
#[derive(Debug, Clone, Default)]
pub struct CanonicalAsrResponse { pub text: String }
#[derive(Debug, Clone, Default)]
pub struct CanonicalRerankResponse { pub scores: Vec<f32> }
#[derive(Debug, Clone, Default)]
pub struct CanonicalEmbeddingResponse { pub vectors: Vec<Vec<f32>> }
#[derive(Debug, Clone, Default)]
pub struct CanonicalVisionResponse {
    pub model: String,
    pub task: String,
    pub image_size: (u32, u32),
    pub inference_time_ms: u32,
    /// JSON-serialized task-specific payload (detections / classifications / segments / poses)
    pub payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalLlmRequest {
    pub model: String,
    #[serde(default)]
    pub system: Vec<SystemBlock>,
    #[serde(default)]
    pub messages: Vec<CanonicalMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub sampling: SamplingParams,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    pub stream: bool,
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    #[serde(default)]
    pub extensions: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMessage {
    pub role: MessageRole,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalLlmResponse {
    pub id: String,
    pub model: String,
    pub stop_reason: StopReason,
    pub content: Vec<ContentBlock>,
    pub usage: UsageInfo,
    #[serde(default)]
    pub extensions: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub enum CanonicalLlmStreamEvent {
    MessageStart { message: CanonicalLlmResponse },
    ContentBlockStart { index: u32, block: ContentBlock },
    ContentBlockDelta { index: u32, delta: ContentDelta },
    ContentBlockStop { index: u32 },
    MessageDelta { stop_reason: Option<StopReason>, usage: Option<UsageInfo> },
    MessageStop,
    Ping,
    Error { message: String },
}

#[derive(Debug, Clone)]
pub enum ContentDelta {
    TextDelta(String),
    InputJsonDelta(String),
    ReasoningDelta(String),
    SignatureDelta(String),
}

impl CanonicalLlmRequest {
    pub fn all_content_blocks(&self) -> Vec<&ContentBlock> {
        self.messages.iter().flat_map(|m| m.content.iter()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_llm_request_serde_round_trip() {
        let req = CanonicalLlmRequest {
            model: "test-model".into(),
            system: vec![],
            messages: vec![],
            tools: vec![],
            sampling: Default::default(),
            stop_sequences: vec![],
            stream: false,
            cache_control: None,
            metadata: Default::default(),
            extensions: Default::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CanonicalLlmRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "test-model");
        assert_eq!(back.stream, false);
    }

    #[test]
    fn test_canonical_llm_stream_event_variants_exist() {
        use ContentDelta::*;
        let _ev = CanonicalLlmStreamEvent::ContentBlockDelta { index: 0, delta: TextDelta("hi".into()) };
        let _ev = CanonicalLlmStreamEvent::MessageStop;
        let _ev = CanonicalLlmStreamEvent::Error { message: "x".into() };
    }
}