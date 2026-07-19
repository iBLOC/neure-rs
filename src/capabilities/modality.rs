use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Modality {
    TextInput,
    TextOutput,
    ImageInput,
    ImageOutput,
    AudioInput,
    AudioOutput,
    VideoInput,
    VideoOutput,
    DocumentInput,
    EmbeddingOutput,
    ToolInput,
    /// Vision model output: bounding boxes + class names + confidence scores
    /// (used for YOLO-style object detection).
    BoundingBox,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextInput => "text_input",
            Self::TextOutput => "text_output",
            Self::ImageInput => "image_input",
            Self::ImageOutput => "image_output",
            Self::AudioInput => "audio_input",
            Self::AudioOutput => "audio_output",
            Self::VideoInput => "video_input",
            Self::VideoOutput => "video_output",
            Self::DocumentInput => "document_input",
            Self::EmbeddingOutput => "embedding_output",
            Self::ToolInput => "tool_input",
            Self::BoundingBox => "bounding_box",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_as_str_round_trip() {
        for m in [
            Modality::TextInput, Modality::TextOutput,
            Modality::ImageInput, Modality::ImageOutput,
            Modality::AudioInput, Modality::AudioOutput,
            Modality::VideoInput, Modality::VideoOutput,
            Modality::DocumentInput, Modality::EmbeddingOutput,
            Modality::ToolInput,
        ] {
            let s = m.as_str();
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_modality_serde_round_trip() {
        let m = Modality::ToolInput;
        let json = serde_json::to_string(&m).unwrap();
        let back: Modality = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }
}