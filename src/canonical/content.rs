use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::capabilities::Modality;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    Audio(AudioBlock),
    Video(VideoBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    pub source: ImageSource,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url(String),
    FileId(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBlock { pub source: AudioSource }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AudioSource {
    Base64 { media_type: String, data: String },
    Url(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoBlock { pub source: VideoSource }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoSource {
    Base64 { media_type: String, data: String },
    Url(String),
    Frames(Vec<ImageSource>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    pub source: ImageSource,
    pub filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Vec<ContentBlock>,
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningBlock {
    pub text: String,
    pub signature: Option<String>,
}

impl ContentBlock {
    pub fn input_modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::TextInput,
            Self::Image(_) => Modality::ImageInput,
            Self::Audio(_) => Modality::AudioInput,
            Self::Video(_) => Modality::VideoInput,
            Self::Document(_) => Modality::DocumentInput,
            Self::ToolUse(_) => Modality::ToolInput,
            Self::ToolResult(_) => Modality::ToolInput,
            Self::Reasoning(_) => Modality::TextInput,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_modality_per_variant() {
        assert_eq!(ContentBlock::Text(TextBlock { text: "hi".into() }).input_modality(), Modality::TextInput);
        assert_eq!(
            ContentBlock::Image(ImageBlock { source: ImageSource::Url("u".into()), detail: None })
                .input_modality(),
            Modality::ImageInput
        );
        assert_eq!(
            ContentBlock::ToolUse(ToolUseBlock { id: "x".into(), name: "f".into(), input: Value::Null })
                .input_modality(),
            Modality::ToolInput
        );
        assert_eq!(
            ContentBlock::ToolResult(ToolResultBlock {
                tool_use_id: "x".into(),
                content: vec![],
                is_error: None,
            }).input_modality(),
            Modality::ToolInput
        );
    }
}