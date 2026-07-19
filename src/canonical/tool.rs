use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_definition_serde() {
        let t = ToolDefinition {
            name: "get_weather".into(),
            description: Some("Get current weather".into()),
            input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "get_weather");
        assert_eq!(back.input_schema["properties"]["city"]["type"], "string");
    }
}