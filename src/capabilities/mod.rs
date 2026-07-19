//! Capability discriminator: Llm / Tts / Asr / Rerank / Embedding.
//!
//! Re-exported from `crate::models::registry::EngineType` (existing type).
//! The plugin architecture uses this as the dispatch key for both
//! adapters and engine registries.

pub use crate::models::registry::EngineType as Capability;

pub mod modality;

pub use modality::Modality;

pub mod api_style;
pub use api_style::ApiStyle;

pub mod model_caps;
pub use model_caps::ModelCapabilities;

pub mod catalog;
pub use catalog::CapabilityCatalog;

#[derive(Debug, Clone, Default)]
pub struct AdapterCapabilities {
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_prompt_caching: bool,
    pub supports_extended_thinking: bool,
}