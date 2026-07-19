//! API adapter trait. Each adapter handles one wire format (OpenAI,
//! Anthropic, custom) and converts between the wire format and
//! Canonical types. Adapters are dynamically registered with
//! AdapterRegistry; the router adds them as routes at startup.

use async_trait::async_trait;
use axum::http::HeaderMap;
use bytes::Bytes;

use crate::api_error::ApiResult;
use crate::canonical::{CanonicalRequest, CanonicalResponse, CanonicalStreamEvent};
use crate::capabilities::AdapterCapabilities;

pub mod registry;
pub mod openai_chat;
pub mod anthropic_messages;

pub use registry::AdapterRegistry;
pub use openai_chat::OpenAiChatAdapter;
pub use anthropic_messages::AnthropicMessagesAdapter;

#[async_trait]
pub trait ApiAdapter: Send + Sync {
    fn name(&self) -> &str;

    fn paths(&self) -> &[&'static str];

    fn parse(&self, body: &Bytes, headers: &HeaderMap) -> ApiResult<CanonicalRequest>;

    fn serialize_response(&self, resp: &CanonicalResponse) -> ApiResult<Bytes>;

    fn serialize_stream_event(&self, event: &CanonicalStreamEvent) -> ApiResult<Option<Bytes>>;

    fn response_content_type(&self) -> &'static str;

    fn stream_content_type(&self) -> &'static str;

    fn capabilities(&self) -> AdapterCapabilities;
}