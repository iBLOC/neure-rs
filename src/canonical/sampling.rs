use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub thinking_budget: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    pub cache_type: CacheType,
    pub ttl: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    Ephemeral,
    Persistent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sampling() {
        let s = SamplingParams::default();
        assert_eq!(s.temperature, None);
        assert!(s.stop_sequences.is_empty());
    }

    #[test]
    fn test_sampling_serde() {
        let s = SamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: Some(256),
            thinking_budget: Some(1024),
            stop_sequences: vec!["END".into()],
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"thinking_budget\":1024"));
        let back: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.temperature, Some(0.7));
    }
}