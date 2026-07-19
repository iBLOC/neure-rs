use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case", content = "0")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    Refusal,
    Other(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    #[serde(default)]
    pub extensions: HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_reason_serde() {
        let r = StopReason::Other("custom".into());
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"reason\":\"other\""));
        assert!(json.contains("custom"));
    }

    #[test]
    fn test_usage_default() {
        let u = UsageInfo::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.cache_read_tokens, None);
    }
}